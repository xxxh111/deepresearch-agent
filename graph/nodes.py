"""图节点。

节点契约（你在 04_node_state.ipynb 里已经验证过）：
    入参拿到的是**完整状态**，返回值是**部分更新** —— 只返回你要改的 key，
    没返回的 key 保持原值不动。
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import Overwrite, interrupt

from graph.state import OverAllState
from model.factory import get_chat_model
from schemas.research import Evidence, GapVerdict, ResearchPlan
from utils.console import console, emit
from utils.prompt_loader import load_prompt
from utils.report import append_warning, save_report, validate_citations



def planner_node(state: OverAllState) -> dict:
    """把研究主题拆成若干可独立检索的子问题。

    M2 实现：真实的 LLM 调用 + 结构化输出。
    模型直接产出 ResearchPlan 对象，全程没有 json.loads、没有正则切分。
    """
    topic = state.get("topic", "").strip()
    if not topic:
        raise ValueError("state 里没有 topic —— 入口节点必须先把研究主题写进状态")

    # 先发进度，再调模型。用户在等模型返回的那几秒能看到反馈。
    emit("正在制定研究计划…")

    # ---- 组装提示词 ----
    # 提示词文件只负责「怎么思考」，格式由下面的 ResearchPlan schema 约束。
    # 注意 system 是静态的、human 是变量 —— 这个顺序对 DeepSeek 的前缀缓存友好。
    #
    # 被打回重做时，把用户的意见一起带上 —— 否则 planner 只会生成
    # 一模一样的计划，用户会觉得「我说了它根本没听」。
    feedback = (state.get("feedback") or "").strip()
    if feedback:
        human_template = (
            "研究问题：{topic}\n\n"
            "上一版研究计划被用户打回了。用户的意见是：\n"
            "{feedback}\n\n"
            "请根据这个意见重新规划，不要重复上一版的拆解思路。"
        )
    else:
        human_template = "研究问题：{topic}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("planner")),
            ("human", human_template),
        ]
    )

    # ---- 组装链 ----
    # 这里是 LCEL：prompt | llm，数据从左边流到右边。
    #
    # with_structured_output 收的是**类**（ResearchPlan），不是实例。
    # 它内部做两件事：
    #   1. ResearchPlan.model_json_schema() → 把类转成 JSON Schema
    #   2. 把这个 schema 伪装成一个工具注册给模型，模型按 schema 填参数
    # 传实例进去的话它拿不到 schema，会直接报错。
    llm = get_chat_model("planner").with_structured_output(ResearchPlan)

    # 注意：链在节点**内部**构建，不在模块顶层。
    # 顶层构建的话，import graph.nodes 就会真的去创建模型客户端 ——
    # 缺 API key 时连 check_env 和单元测试都跑不起来。
    # （这就是 model/factory.py 做成惰性构造的原因，这里配合它。）
    chain = prompt | llm

    # 只传模板里真正用到的变量 —— 多传未使用的变量有些版本会报警告
    inputs = {"topic": topic}
    if feedback:
        inputs["feedback"] = feedback

    plan = chain.invoke(inputs)

    # 返回部分更新：只写 plan，不碰 topic
    return {"plan": plan}


def human_review_node(state: OverAllState) -> dict:
    """暂停图，请用户审核研究计划。

    ============================ interrupt() 做了什么 ============================

    1. 抛出一个特殊异常，让图**在这里停下来**，把当前状态写进检查点
    2. 把 payload 交出去，让外面的人看到「这里需要人做个决定」

    外面的人（CLI / UI）拿到 payload → 问用户 → 然后：

        graph.invoke(Command(resume=用户的回答), config)

    图就从这里继续往下走。

    ==================== ⚠️ 最反直觉的一点：这段代码会重跑 ====================

    resume 的时候，LangGraph 是**重新执行整个 human_review_node**，
    而不是「从 interrupt() 那一行接着往下走」。

    所以 interrupt() **之前**的代码会跑第二遍：

        · print          → 用户看到重复输出
        · 写文件          → 文件被写两次
        · 调用计费 API    → 钱花两遍

    规则：**interrupt() 之前只做纯计算，任何副作用都放到它后面。**

    这不是 bug，是设计 —— 节点必须可重放，
    否则 LangGraph 无法保证「恢复后的状态和中断前完全一致」。

    =============================================================================
    """
    plan = state.get("plan")
    if plan is None:
        raise ValueError("没有研究计划可以审核 —— planner 节点必须先跑")

    # ★ 图在这一行停下。payload 会原样交出去给 CLI/UI 展示。
    #   注意上面只做了取值和判空 —— 没有任何副作用，所以重跑它是安全的。
    answer = interrupt(
        {
            "type": "plan_review",
            "topic": state.get("topic", ""),
            "rationale": plan.rationale,
            "sub_questions": list(plan.sub_questions),
        }
    )

    # ↓↓↓ 下面的代码只在用户回答**之后**才执行 ↓↓↓
    raw = str(answer or "").strip()

    # --- 情况一：打回重做。以 "no" 开头，后面的文字当反馈 ---
    if raw.lower() == "no" or raw.lower().startswith("no "):
        feedback = raw[2:].strip()
        emit("计划已打回，正在重新规划…")
        return {"approved": False, "feedback": feedback or "用户要求重新规划"}

    # --- 情况二：通过。回车直接过；写了逗号分隔的内容就替换子问题 ---
    edits = [s.strip() for s in raw.replace("，", ",").split(",") if s.strip()]
    if edits:
        # 注意：model_copy(update=...) **不会**重新验证 schema，
        # 所以用户可以给出超出 3-5 条的计划 —— 这是有意为之。
        # 人工审核的整个意义就在于「人说了算」，这时再拿 schema 拦住用户
        # 就本末倒置了。（对面是 LLM 时才需要 schema 当护栏。）
        plan = plan.model_copy(update={"sub_questions": edits})
        emit(f"已按你的修改更新计划（{len(edits)} 条子问题）")
    else:
        emit("计划已通过")

    return {"plan": plan, "approved": True, "feedback": ""}


def aggregator_node(state: OverAllState) -> dict:
    """汇总并行分支收集到的证据：给每个来源分配引用编号。

    ============================ 为什么这件事必须在这里做 ============================
    N 个 researcher 分支是**并发**的，每个分支都不知道：
      · 自己是第几个跑完的
      · 别的分支读到了哪些页面

    所以「全局编号」这类需要看到**全部结果**的操作，只能放在下游这个
    唯一的、确定性的节点里做。这叫「把确定性逻辑从并发中拿出来」。

    ===============================================================================

    两点实现上的讲究（都是踩坑踩出来的）：

    ① 用 dict 而不是 set 来登记来源
       set 的遍历顺序依赖字符串哈希，而 Python 从 3.3 起对字符串哈希做
       **进程级随机化**。实测同一个集合三次运行得到三种顺序 ——
       编号会跟着变，报告不可复现。
       普通 dict 保持插入顺序（3.7+），边遍历边分配就天然是「按首次出现顺序」。

    ② 返回值必须用 Overwrite 包起来
       evidence 字段挂了 operator.add reducer，直接返回 {"evidence": [...]}
       是**追加**而不是替换，列表会翻倍（实测 3 条变 6 条）。
       Overwrite 的作用就是绕过 reducer 强制替换 —— 这正是它被设计出来解决的问题。
    """
    evidence = state.get("evidence") or []
    if not evidence:
        emit("没有收到任何证据")
        return {}

    # 来源 URL → 引用编号。边遍历边分配，天然按首次出现顺序。
    url_to_citation: dict[str, int] = {}
    numbered: list[Evidence] = []

    for item in evidence:
        if item.source_url not in url_to_citation:
            url_to_citation[item.source_url] = len(url_to_citation) + 1
        numbered.append(
            item.model_copy(
                update={"citation_index": url_to_citation[item.source_url]}
            )
        )

    iteration = state.get("iteration", 0) + 1

    emit(
        f"汇总（第 {iteration} 轮）：{len(evidence)} 条证据来自 "
        f"{len(url_to_citation)} 个页面，已分配引用编号"
    )

    return {
        # ★ Overwrite：强制替换，而不是往 reducer 里追加
        "evidence": Overwrite(numbered),
        # 轮次计数放在这里，因为 aggregator 每轮**恰好执行一次**
        # （不管那一轮有几个并行分支）。放在分支里会数成「分支数」，
        # 放在 gap_checker 里会数成「检查次数」—— 都不对。
        "iteration": iteration,
    }


# ============================================================================
# M7：自我反思 → 报告 → 追问
# ============================================================================
@lru_cache(maxsize=1)
def _gap_chain():
    return ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("gap_checker")),
            (
                "human",
                "研究主题：{topic}\n\n"
                "原本规划的子问题：\n{sub_questions}\n\n"
                "目前已收集到的证据概览：\n{evidence_digest}",
            ),
        ]
    ) | get_chat_model("gap_checker").with_structured_output(GapVerdict)


def _evidence_digest(evidence: list[Evidence], plan: ResearchPlan | None) -> str:
    """把证据压成「给审查员看的概览」。

    【为什么不直接把完整证据喂进去】
    写报告需要原文引用，判断「够不够」不需要 —— 审查员看的是
    「每个子问题下有几条什么结论」，不是逐字核对来源。
    完整证据现在有 70+ 条，全塞进去光这一项就是上万 token，
    而它换来的判断质量提升接近于零。

    **按用途裁剪输入**，是控制成本最有效也最容易被忽略的手段。
    """
    if not evidence:
        return "（一条证据都没有）"

    # 按子问题分组
    grouped: dict[str, list[Evidence]] = {}
    for item in evidence:
        grouped.setdefault(item.sub_question or "(未标注子问题)", []).append(item)

    lines: list[str] = []
    for question, items in grouped.items():
        high = sum(1 for i in items if i.confidence == "high")
        lines.append(f"\n【子问题】{question}")
        lines.append(f"  证据 {len(items)} 条（其中高可信 {high} 条）：")
        for item in items[:12]:  # 单个子问题最多列 12 条，防止某一条撑爆
            lines.append(f"    - {item.claim[:120]}")
        if len(items) > 12:
            lines.append(f"    …… 另有 {len(items) - 12} 条")

    # 把规划过、但一条证据都没收到的子问题显式标出来 ——
    # 这是审查员最需要看到的信息
    if plan is not None:
        covered = set(grouped)
        for question in plan.sub_questions:
            if not any(question[:20] in c for c in covered):
                lines.append(f"\n【子问题】{question}")
                lines.append("  ⚠️ 没有任何证据")

    return "\n".join(lines)


def gap_checker_node(state: OverAllState) -> dict:
    """判断现有证据是否足以写报告；不足则给出补充检索词。

    这是整个项目里「自我反思」的实现 —— 也是多智能体相对于单次 RAG
    的核心差异：它知道自己知道什么、不知道什么。
    """
    evidence = state.get("evidence") or []
    iteration = state.get("iteration", 1)

    emit(f"正在检查证据是否充分（第 {iteration} 轮）…")

    verdict = _gap_chain().invoke(
        {
            "topic": state.get("topic", ""),
            "sub_questions": "\n".join(
                f"  - {q}" for q in (state.get("plan").sub_questions if state.get("plan") else [])
            ),
            "evidence_digest": _evidence_digest(evidence, state.get("plan")),
        }
    )

    if verdict.sufficient:
        emit("证据充分，开始撰写报告")
    else:
        gaps = "；".join(verdict.gaps[:3]) or verdict.reason
        emit(f"证据不足：{gaps}")

    return {"gap_verdict": verdict}


@lru_cache(maxsize=1)
def _writer_chain():
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("writer")),
                (
                    "human",
                    "研究主题：{topic}\n\n"
                    "可用证据（编号是引用时要用的）：\n{evidence_table}\n\n"
                    "{gap_note}"
                    "请撰写研究报告。",
                ),
            ]
        )
        | get_chat_model("writer")
        # ★ StrOutputParser 必须加 —— 这一步在 M7 踩了坑。
        #
        # 不加的话，chain.stream() 产出的是 **AIMessageChunk 对象**，
        # 不是字符串。后果：
        #   · "".join(pieces) 拼不出来
        #   · 想推给前端流式输出时直接 TypeError: write() argument must be str
        #
        # 记住这条：**链的最后一步决定了你拿到什么类型**。
        # prompt | llm            →  AIMessage / AIMessageChunk
        # prompt | llm | parser   →  str
        | StrOutputParser()
    )


def _evidence_table(evidence: list[Evidence]) -> str:
    """给撰写者的证据表。

    【为什么这里不给 quote】
    writer 的任务是「组织论述 + 标注引用」，不是「逐字引用原文」。
    读者拿 [n] 去文末找 URL 再自己核对 —— quote 是给读者验证用的，
    不是 writer 的必需品。省下这部分能让提示词小一大截。
    （如果以后发现报告内容失真，再考虑把 quote 加回来。）
    """
    lines = []
    for item in sorted(evidence, key=lambda e: e.citation_index or 0):
        if not item.citation_index:
            continue
        lines.append(
            f"[{item.citation_index}] {item.claim}"
            f"  —— {item.source_title} (可信度 {item.confidence})"
        )
    return "\n".join(lines)


def writer_node(state: OverAllState) -> dict:
    """写带引用的 Markdown 报告，并**流式**推给前端。

    这是全项目唯一一个「边生成边输出」的节点 ——
    报告有两千多字，等它整篇写完再显示，用户要干等十几秒。
    """
    emit("正在撰写报告…")

    evidence = state.get("evidence") or []
    verdict = state.get("gap_verdict")

    gap_note = ""
    if verdict is not None and not verdict.sufficient:
        gaps = "；".join(verdict.gaps) or verdict.reason
        gap_note = (
            f"【重要】补充检索后仍有未解决的信息缺口：{gaps}\n"
            "请在报告末尾用「## 局限与未决问题」一节明确说明，"
            "不要在正文里用推测填补。\n\n"
        )

    pieces: list[str] = []
    for delta in _writer_chain().stream(
        {
            "topic": state.get("topic", ""),
            "evidence_table": _evidence_table(evidence),
            "gap_note": gap_note,
        }
    ):
        pieces.append(delta)
        _emit_report_delta(delta)

    report = "".join(pieces).strip()

    # ---- 生成后立刻做机器校验：报告里的 [n] 是否都真实存在 ----
    result = validate_citations(report, evidence)

    if result["missing"]:
        msg = (
            f"报告引用了不存在的编号 {result['missing']}，"
            "已在报告末尾标注。"
        )
        console.print(f"  [bold red]⚠ {msg}[/bold red]")
        report = append_warning(
            report,
            f"本报告引用了编号 {result['missing']}，但证据表中没有对应来源。"
            "请谨慎对待这些内容。",
        )
    if result["unused"]:
        console.print(
            f"  [yellow]提示：编号 {result['unused']} 的证据没有被引用[/yellow]"
        )

    # ---- 落盘 ----
    # thread_id 不在业务状态里（它是运行时配置），所以从 config 读。
    # 这样做的好处：状态 schema 保持干净，只装业务数据。
    from langgraph.config import get_config

    thread_id = (get_config().get("configurable") or {}).get("thread_id", "unknown")
    path = save_report(state.get("topic", "未命名研究"), report, thread_id)

    emit(f"报告已完成，共 {len(report)} 字")
    return {"report": report, "report_path": str(path)}


def _emit_report_delta(text) -> None:
    """把报告正文的增量推给前端。

    用一个带 kind 的字典和普通进度消息区分开 ——
    CLI 收到 report_delta 就原样输出（不打前缀），
    收到别的一律当成进度提示。
    """
    if not text:
        return

    # 防御：链的末尾如果漏了 StrOutputParser，这里拿到的是 AIMessageChunk。
    # 根因已经在 _writer_chain 里修了，但这道兜底留着 ——
    # 「流式输出」这种跨模块的类型契约，最容易被后来的改动悄悄破坏，
    # 而报错会出现在很远的地方（CLI 的 write() 里），很难定位。
    if not isinstance(text, str):
        text = getattr(text, "content", "") or ""

    if not text:
        return

    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"kind": "report_delta", "text": text})
    except Exception:
        pass


@lru_cache(maxsize=1)
def _qa_chain():
    return ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("qa_followup")),
            (
                "human",
                "研究主题：{topic}\n\n"
                "已完成的报告：\n{report}\n\n"
                "可引用的证据表：\n{evidence_table}",
            ),
            MessagesPlaceholder("messages"),
        ]
    ) | get_chat_model("qa")


def qa_followup_node(state: OverAllState) -> dict:
    """在同一份研究材料上回答追问，**不触发任何检索**。

    这是检查点带来的直接收益：检索结果都还在状态里，
    追问只需要一次 LLM 调用，零搜索额度。
    """
    question = state.get("followup_question", "")
    emit(f"正在基于既有材料回答：{question[:40]}…")

    answer = _qa_chain().invoke(
        {
            "topic": state.get("topic", ""),
            "report": state.get("report", "（报告尚未生成）"),
            "evidence_table": _evidence_table(state.get("evidence") or []),
            "messages": [
                *(state.get("messages") or []),
                HumanMessage(content=question),
            ],
        }
    )

    # ⚠️ 只写 messages，**不碰 report**。
    #
    # 追问的回答是对原报告的补充说明，不是新报告。如果写进 report 字段，
    # 用户上一次辛苦跑出来的报告就被这一次追问覆盖了 ——
    # 而且检查点里也一起没了（这正是为什么要先想清楚「哪个字段归谁写」）。
    #
    # 这也顺带说明了一件重要的事：
    # **同一份状态上做多个功能时，字段的所有权必须明确。**
    return {"messages": [HumanMessage(content=question), answer]}


# ============================================================================
# 关于 with_structured_output 的 method 参数
# ============================================================================
# 这里没传 method=，走默认值 function_calling —— 这正是我们要的：
# 借 tool calling 机制拿到强类型对象。
#
# 什么时候才需要显式传：
#   method="json_mode"  —— 模型不支持 tool calling，或 function_calling 反复失败时。
#                          注意 json_mode 只保证「返回合法 JSON」，不保证符合 schema，
#                          而且提示词里必须出现「JSON」字样，否则 DeepSeek 会拒绝。
#   method="json_schema" —— 模型原生支持严格 schema 约束（如 OpenAI 的 structured outputs）
#
# 所以降级链是： function_calling（默认） → 失败再考虑 json_mode
#
# ============================================================================
# 如果模型反复不遵守 min_length=3 怎么办？
# ============================================================================
# 现在：Pydantic 验证失败 → 抛异常 → RetryPolicy 重试 3 次（任务 3）
# 三次都失败的话，异常会冒到图外面，整个运行终止。
#
# 三种可选思路，各有取舍 —— 想清楚再动手：
#   1. 放宽 schema（min_length=2），让提示词去保证质量。牺牲约束换稳定性
#   2. 加一个「补全」节点：条数不够就再调一次模型补齐。
#      多一次调用成本，但能救回一次完整运行
#   3. 接受失败。规划都做不好的问题，搜出来的报告大概也好不了，
#      早点失败比拖到最后浪费一堆检索额度更划算
#
# 面试被问到时，说出你的取舍理由比说出方案本身更重要。
