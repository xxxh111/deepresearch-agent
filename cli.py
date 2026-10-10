"""DeepResearch Agent —— 命令行入口。

用法：
    conda activate langgraph
    python cli.py "为什么多智能体架构比单智能体更适合深度研究任务"

M1 阶段：跑通 planner 一个节点，验证「接线」无误。
后续里程碑会逐步长成完整的并行 + 人工审核 + 带引用的报告流程。
"""

import argparse
import sys
import textwrap
import uuid

import utils.console  # noqa: F401  —— 必须最先导入：全进程修复 Windows 中文编码
from langgraph.types import Command

from utils.config_handler import research_conf
from utils.console import console, rule, sanitize_text
from utils.tracing import setup_tracing


def _unpack(chunk):
    """兼容 subgraphs 开关带来的 chunk 形状差异。

    不加 subgraphs=True：  chunk 是 (mode, data) 二元组
    加了 subgraphs=True：  chunk 是 (namespace, mode, data) 三元组
                            namespace 形如 ("researcher:<task_id>",)

    这是 LangGraph 流式最常见的语法坑，所以提前写成兼容形式，
    M5 打开 subgraphs 时这里不用改。
    """
    if len(chunk) == 3:
        namespace, mode, data = chunk
        return namespace, mode, data
    mode, data = chunk
    return (), mode, data


def run(
    topic: str | None = None,
    thread_id: str = "",
    resume: bool = False,
    ask: str | None = None,
) -> None:
    # 延迟导入：确保 utils.console 的编码修复先于任何图相关模块生效
    from graph import build_graph

    graph = build_graph()

    # thread_id 放在 configurable 里 —— 它是「存档名」，
    # 检查点靠它区分不同的执行历史。
    config = {
        "configurable": {"thread_id": thread_id},
        # recursion_limit 是 LangGraph 层面的硬顶，防止任何意外的边循环把进程跑死。
        # M7 长出循环边之后，这道闸门就变得至关重要。
        "recursion_limit": research_conf["recursion_limit"],
    }

    # 三种输入形态，对应三种完全不同的行为：
    #
    #   {"topic": ...}            新研究。从头跑 planner → 检索 → 报告
    #   {"followup_question": ...} 追问。入口路由会跳过 planner 和全部检索，
    #                              直接用检查点里已有的证据和报告回答
    #   None                      续跑。不要新输入，从检查点接着执行
    #
    # ★ 追问也要带上 thread_id —— 正是靠它才找得到上一次的研究材料。
    #   这也是「一份研究反复使用」这个卖点的技术基础。
    if ask:
        payload: dict | None = {"followup_question": ask}
    elif resume:
        payload = None
    else:
        payload = {"topic": topic}

    if ask:
        saved = graph.get_state(config)
        if not saved.values.get("report"):
            console.print(
                f"[red]thread_id={thread_id!r} 上没有已完成的报告，无法追问。[/red]"
            )
            console.print("[dim]先跑一次完整研究，再用 --ask 追问。[/dim]")
            return
        rule(f"追问：{ask}")
        console.print(f"[dim]thread_id = {thread_id}[/dim]")
        console.print(
            f"[dim]复用已有材料：{len(saved.values.get('evidence') or [])} 条证据，"
            "本次**不触发任何检索**[/dim]"
        )

    elif resume:
        saved = graph.get_state(config)
        if not saved.values:
            console.print(f"[red]thread_id={thread_id!r} 没有任何检查点。[/red]")
            console.print("[dim]是不是 config/research.yml 里用的是 memory？那退出即忘。[/dim]")
            return

        rule(f"从检查点恢复：{saved.values.get('topic', '(未知主题)')}")
        console.print(f"[dim]thread_id = {thread_id}[/dim]")

        if saved.interrupts:
            # ★ 跨进程恢复人工审核：上一次进程停在 interrupt 上就退出了，
            #   现在换个进程、甚至隔一天，照样能从存档里读出待办的审核。
            #   这是 interrupt + checkpointer 组合起来最有价值的场景 ——
            #   生产环境里审批可能要等几小时甚至几天，
            #   总不能让那个进程一直挂着。
            console.print("[yellow]发现一个待处理的人工审核。[/yellow]")
            payload = Command(resume=_ask_plan_review(saved.interrupts[0]))
        elif saved.next:
            console.print(
                f"[dim]检查点快照：{len(list(graph.get_state_history(config)))} 个"
                "（每个超步存一次）[/dim]"
            )
            console.print(f"[dim]待执行节点：{saved.next}[/dim]")
            payload = None
        else:
            console.print("[yellow]这条历史已经跑完了，没有待执行的节点。[/yellow]")
            return
    else:
        rule(f"研究主题：{topic}")
        console.print(f"[dim]thread_id = {thread_id}[/dim]")

    console.print()

    try:
        # 循环处理中断：用户打回计划 → 图回到 planner → 再次审核 → 又中断一次。
        # 所以不能只处理一次中断，要一直处理到图真正跑完为止。
        while True:
            pending = _stream(graph, payload, config)
            if pending is None:
                break
            payload = Command(resume=_ask_plan_review(pending))
    except Exception as exc:
        _report_failure(exc)
        raise

    # ★ 渲染一律以**检查点里的最终状态**为准，而不是这次流里看到的增量。
    #
    # 为什么：恢复运行时，planner 节点早在上一次进程里就跑完了，
    # 这次的 updates 流里根本不会出现它 —— 用增量渲染就会显示「没有拿到计划」。
    # 检查点才是权威状态：不管这次跑了几个节点，它里面永远是完整的。
    _render(dict(graph.get_state(config).values), thread_id)


def _stream(graph, payload, config: dict):
    """跑图、打印实时进度。

    返回本次运行遇到的中断（Interrupt 对象）；没有中断则返回 None。
    最终状态由调用方从检查点读取，不在这里收集。
    """
    # 本函数只负责「打印进度」和「让图跑完」。
    # 最终状态的读取交给调用方（从检查点拿），不在这里收集增量 ——
    # 恢复运行时增量是不完整的（已完成的节点不会再出现在流里）。
    pending = None

    # ★ M5 打开 subgraphs=True
    #
    # 不加这个参数的话，**子图内部的事件永远流不出来** ——
    # 5 个并行研究员在干什么，你只能看到一个笼统的「researcher 执行完毕」。
    # 加上之后 chunk 从 (mode, data) 二元组变成 (namespace, mode, data) 三元组，
    # namespace 形如 ("researcher:<task_id>",)。
    # （_unpack 早就写好了兼容，这里不用改解包逻辑）
    for chunk in graph.stream(
        payload, config, stream_mode=["updates", "custom"], subgraphs=True
    ):
        namespace, mode, data = _unpack(chunk)

        if mode == "custom":
            # 来自节点内的 get_stream_writer()(payload)，即实时的进度信息
            _print_progress(namespace, data)

        elif mode == "updates":
            # subgraphs=True 之后，子图**内部**每个节点的更新也会流出来。
            # 那些对用户没意义（是子图的实现细节），过滤掉，
            # 只在顶层打印「哪个节点完成了」。
            if namespace:
                continue

            # 每个节点返回的状态增量
            for node, update in data.items():
                if node == "__interrupt__":
                    # 图在这里暂停了，把中断对象捞出来交给调用方处理。
                    # updates 流里它的形状是 (Interrupt(...),) —— 一个元组，
                    # 因为理论上一次可以有多个中断。
                    items = update if isinstance(update, (list, tuple)) else (update,)
                    pending = items[0] if items else None
                    continue

                # 跳过 LangGraph 的内部节点（__metadata__ 这类）。
                # 它们不是我们图里的东西，显示出来只会干扰阅读。
                if node.startswith("__"):
                    continue
                console.print(f"  [green]✓[/green] 节点 [bold]{node}[/bold] 执行完毕")

    return pending


def _ask_plan_review(pending) -> str:
    """把 interrupt 的 payload 渲染给用户，读回他的回答。

    这个函数是「图的暂停点」和「人的决定」之间的桥。
    payload 由 human_review_node 里的 interrupt({...}) 决定，
    返回值会通过 Command(resume=...) 送回那个节点的 answer 变量。

    ⚠️ 用户回答之后，human_review_node 会**从头重跑**，
    所以这里打印的内容在恢复后会再出现一次 —— 这是正常的。
    """
    payload = getattr(pending, "value", pending)
    if not isinstance(payload, dict):
        return str(payload or "")

    console.print()
    rule("[yellow]⏸  请审核研究计划[/yellow]")
    console.print(f"[bold]研究主题：[/bold]{payload.get('topic', '')}\n")
    console.print(f"[bold]拆解思路：[/bold]{_wrap(payload.get('rationale', ''), 72)}\n")
    for index, question in enumerate(payload.get("sub_questions") or [], 1):
        console.print(f"  [bold cyan]{index}.[/bold cyan] {_wrap(question, 72, indent=5)}")

    console.print()
    console.print("  [green]回车[/green]        通过，按这个计划开始检索")
    console.print("  [green]用逗号分隔[/green]  替换成你自己的子问题")
    console.print("                  例：欧洲市场政策, 关税影响, 本地化生产")
    console.print("  [green]no 理由[/green]    打回重做，理由会带给规划者")
    console.print("                  例：no 太偏技术了，我想了解市场和政策")
    console.print()

    # 用户输入要过一道 sanitize ——
    # 编码错位产生的代理字符如果进了图状态，会在写检查点时炸掉整轮运行，
    # 而且报错地点离真正的原因很远（你在 CLI 输错，它却死在 sqlite 序列化里）。
    return sanitize_text(console.input("[bold]你的决定 > [/bold]").strip())


def _print_progress(namespace: tuple, data) -> None:
    """处理节点内 get_stream_writer 推出来的两类消息。

    协议很简单：
        {"kind": "report_delta", "text": ...}  → 报告正文的增量，原样输出
        其他（字符串）                          → 进度提示，加前缀和缩进

    为什么报告正文要原样输出？因为它是 Markdown 正文，逐字拼起来才是报告。
    如果每段都加个「·」前缀，复制出来的报告就没法看了。
    """
    if isinstance(data, dict) and data.get("kind") == "report_delta":
        sys.stdout.write(data.get("text", ""))
        sys.stdout.flush()
        return

    # 子图内部的进度：用 task_id 前 4 位当短标签区分并行分支
    if namespace:
        tag = str(namespace[0]).split(":")[-1][:4]
        console.print(f"    [dim]\\[{tag}][/dim] [cyan]·[/cyan] {data}")
    else:
        console.print(f"  [cyan]·[/cyan] {data}")


def _report_failure(exc: Exception) -> None:
    """把真正的失败原因挖出来打给用户。

    LangGraph 的异常往往被包了好几层，traceback 又很长 ——
    关键信息在**最内层的 __cause__**。直接 print(exc) 经常只看到
    「某某节点执行失败」这类没用的话。
    """
    console.print()
    rule("[red]运行失败[/red]")

    root: BaseException = exc
    while root.__cause__ is not None:
        root = root.__cause__

    console.print(f"[bold red]{type(root).__name__}[/bold red]")
    console.print(f"[red]{root}[/red]")

    if root is not exc:
        console.print(f"\n[dim]（外层包装异常：{type(exc).__name__}: {exc}）[/dim]")

    console.print("\n[dim]完整调用栈见上方 traceback。[/dim]")


def _render(state: dict, thread_id: str = "") -> None:
    """渲染最终结果。

    两种形态：
        · 追问 —— 只有一句回答（报告和证据都是上一次的，不重复显示）
        · 研究 —— 研究计划 + 证据清单 + 报告落盘位置
    """
    # ---------- 追问模式 ----------
    if state.get("followup_question"):
        messages = state.get("messages") or []
        console.print()
        rule("回答")
        if messages:
            content = getattr(messages[-1], "content", str(messages[-1]))
            console.print(content)
        else:
            console.print("[yellow]没有拿到回答。[/yellow]")
        return

    # ---------- 研究模式 ----------
    plan = state.get("plan")
    evidence = list(state.get("evidence") or [])

    console.print()
    rule("研究计划")

    if plan is None:
        console.print("[red]没有拿到计划 —— plan 节点没有写回状态。[/red]")
        return

    console.print(f"[bold]拆解思路：[/bold]{_wrap(plan.rationale, 74)}\n")
    for i, question in enumerate(plan.sub_questions, 1):
        console.print(f"  [bold cyan]{i}.[/bold cyan] {_wrap(question, 74, indent=5)}")

    if not evidence:
        console.print("\n[yellow]没有拿到任何证据。[/yellow]")
        return

    sources = {item.source_url for item in evidence}
    rule(f"证据（{len(evidence)} 条，来自 {len(sources)} 个页面）")

    for item in evidence:
        # citation_index 由 aggregator 赋值（如果那一版已经实现了的话）
        tag = f"[{item.citation_index}]" if item.citation_index else "·"
        console.print(f"\n  [bold cyan]{tag}[/bold cyan] {_wrap(item.claim, 68, indent=5)}")
        console.print(f"     引用：{_wrap(item.quote, 66, indent=10)}")
        console.print(f"     [dim]来源：{item.source_title[:56]}[/dim]")
        console.print(f"     [dim]可信度：{item.confidence}[/dim]")

    # ---------- 报告 ----------
    # 报告正文已经在 writer 节点里**流式**打过了，这里不重复显示，
    # 只给统计和落盘位置。
    report = state.get("report") or ""
    verdict = state.get("gap_verdict")

    console.print()
    rule("完成")

    if not report:
        console.print("[yellow]没有生成报告。[/yellow]")
        return

    console.print(f"  证据      : {len(evidence)} 条，来自 {len(sources)} 个页面")
    console.print(f"  研究报告  : {len(report)} 字")
    console.print(f"  研究轮次  : {state.get('iteration', '?')} 轮")

    if verdict is not None and not verdict.sufficient:
        # 诚实降级：到迭代上限仍有缺口时，明确告诉用户，别让人以为查全了
        console.print(
            f"  [yellow]遗留缺口: {'；'.join(verdict.gaps) or verdict.reason}[/yellow]"
        )
        console.print("  [dim]（已写入报告的「局限与未决问题」一节）[/dim]")

    path = state.get("report_path")
    if path:
        console.print(f"\n  报告已保存：[bold green]{path}[/bold green]")

    console.print("\n  [dim]追问（复用已有材料，不重新检索）：[/dim]")
    console.print(
        f'  [dim]  python cli.py --thread-id {thread_id} --ask "你的问题"[/dim]'
    )


def _wrap(text: str, width: int, indent: int = 0) -> str:
    """按显示宽度手工折行，续行缩进对齐。

    不交给 rich 自动折行的原因：它从行首开始对齐，
    长问句换行后第二行会顶到最左边，编号和内容分不清。
    """
    return textwrap.fill(
        text,
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    ).lstrip()


def main() -> None:
    # 必须在第一次调用模型之前执行。
    # 否则 .env 里的 LANGSMITH_TRACING=true 会生效，而 API key 还是占位符，
    # langsmith 就会拿着一串 lsv2_xxxx 去上报，然后在流程结束时甩一个 403 出来 ——
    # 图跑得好好的，末尾却报错，很误导人。
    # setup_tracing() 会检测占位符并自动关掉 tracing。
    setup_tracing()

    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="DeepResearch Agent —— 基于 LangGraph 的多智能体深度研究助手",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="研究主题，用引号包起来。用 --resume 时可以省略",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help=(
            "会话 ID（相当于存档名）。同一个 ID 就是同一条执行历史，"
            "用它做断点续跑和多轮追问。不传则每次随机生成"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "从上次中断的地方继续。必须配合 --thread-id 使用，"
            "且不要传 topic —— 图会从检查点里恢复，而不是接受新输入"
        ),
    )
    parser.add_argument(
        "--ask",
        default=None,
        help=(
            "基于已有报告追问。必须配合 --thread-id 使用。"
            "追问**不会触发任何检索** —— 直接用存档里的证据和报告回答"
        ),
    )
    args = parser.parse_args()

    # ---------- 参数校验 ----------
    modes = [bool(args.resume), bool(args.ask), bool(args.topic)]
    if sum(modes) > 1:
        parser.error(
            "topic / --resume / --ask 只能三选一：\n"
            "  topic       新研究\n"
            "  --resume    接着上次没跑完的继续\n"
            "  --ask       基于已完成的报告追问"
        )

    if args.resume and not args.thread_id:
        parser.error("--resume 必须配合 --thread-id 使用，否则不知道该恢复哪条历史")

    if args.ask and not args.thread_id:
        parser.error(
            "--ask 必须配合 --thread-id 使用，否则不知道要基于哪一份报告回答"
        )

    if not any(modes):
        parser.error("请提供研究主题；或者用 --resume / --ask 接着上次的会话")

    # 命令行参数同样过一道 sanitize。
    # sys.argv 在 Windows 上按文件系统编码解码，中文参数在某些终端下
    # 也可能带出代理字符 —— 同一个坑，不同的入口。
    thread_id = sanitize_text(args.thread_id) if args.thread_id else f"run-{uuid.uuid4().hex[:8]}"
    topic = sanitize_text(args.topic) if args.topic else None
    ask = sanitize_text(args.ask) if args.ask else None

    try:
        run(topic=topic, thread_id=thread_id, resume=args.resume, ask=ask)
    except KeyboardInterrupt:
        _on_interrupt(thread_id)


def _on_interrupt(thread_id: str) -> None:
    """Ctrl+C 之后告诉用户怎么接着跑。

    这是检查点机制最直观的价值体现：中断不等于前功尽弃 ——
    前面已经烧掉的检索额度都还存在检查点里。
    """
    console.print()
    rule("[yellow]已中断[/yellow]")
    console.print("  已完成的工作已经存进检查点了，没有白跑。\n")
    console.print("  继续执行：")
    console.print(f"    [bold green]python cli.py --thread-id {thread_id} --resume[/bold green]\n")
    console.print(
        "  [dim]（前提是 config/research.yml 里 checkpointer: sqlite。"
        "如果是 memory，进程一退状态就没了）[/dim]"
    )


if __name__ == "__main__":
    main()
