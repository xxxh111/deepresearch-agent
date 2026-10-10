"""研究流程的 Pydantic 数据模型。

【为什么用 Pydantic，而不是自己 json.loads / 正则切分】
LLM 输出是不可信文本。用 with_structured_output 把 schema 交给模型之后：
    1. 解析失败会**直接抛异常**，被节点的 RetryPolicy 接住重试 ——
       而不是悄悄产出一个字段缺失的字典继续往下游跑（那种 bug 最难查）
    2. 字段类型有保证，状态里存的是对象不是字符串
    3. LangSmith 里能看到结构化的输入输出，调试成本低很多

【一个必须知道的约束】
结构化输出要求 schema 顶层是**对象**，不能直接是 list。
所以「一组证据」必须包成 EvidenceList(items=[...]) 这种容器模型，
不能写 with_structured_output(list[Evidence])。
"""

from typing import Literal

from pydantic import BaseModel, Field


# ============================================================================
# 规划
# ============================================================================
class ResearchPlan(BaseModel):
    """规划节点的输出：把用户问题拆成若干个可独立检索的子问题。"""

    rationale: str = Field(
        description="为什么这样拆解。一句话说明拆解思路，让人能判断拆得对不对。"
    )

    sub_questions: list[str] = Field(
        description=(
            "3-5 个子问题。每条都必须是**可以独立拿去搜索引擎检索**的完整问句，"
            "彼此不重叠，合起来能覆盖原问题的全貌。"
        ),
        # 说明：这里的上下界是「安全护栏」而不是主要约束。
        # 精确的 3-5 由提示词保证；schema 只负责拦住明显跑偏的情况（比如只给了 1 条）。
        # 如果实测发现 DeepSeek 反复不遵守导致验证失败，把下界放宽到 2。
        min_length=3,
        max_length=5,
    )


# ============================================================================
# 证据 —— 研究和写作之间的契约
# ============================================================================
# ---------------------------------------------------------------------------
# 【为什么这个模型关键 —— 写给读代码的人，不发给模型】
#
# 它是 researcher 和 writer 之间的唯一接口：
#
#     researcher 提炼  ──▶  aggregator 去重编号  ──▶  writer 写报告
#
# writer 能写出什么样的报告，完全取决于 Evidence 里装了什么。
# 字段少了，writer 只能干写结论；字段多了，模型填不准反而引入噪声。
#
# 【类 docstring 会不会发给模型？—— 会。这是个容易踩的坑】
# Pydantic 会拿类 docstring 当整个对象的 description 写进 JSON Schema，
# 而 with_structured_output 会把这个 schema 整个交给模型。
# 所以 docstring 必须写成**给模型看的**，教学笔记用 # 注释写。
# 验证方法：python -m schemas.research，看输出的 JSON Schema 第一个字段。
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    """一条可溯源的证据：一个来自特定网页、且有原文片段支撑的结论。"""

    claim: str = Field(
        description=(
            "从原文中提炼出的一条结论或事实，用完整的句子表述。"
            "必须是原文**明确支持**的内容 —— 不要加入你自己的推断、"
            "常识补充或跨来源的综合判断。一条证据只对应一个来源。"
        )
    )

    quote: str = Field(
        description=(
            "支撑上面这条结论的**原文片段**，从网页内容里逐字摘录。"
            "不要改写、不要翻译、不要拼接不相邻的句子。"
            "**如果找不到可以直接引用的原文片段，就不要产出这条证据。**"
        )
    )

    source_url: str = Field(description="这条证据的来源网页地址，必须是完整 URL")

    source_title: str = Field(description="来源网页的标题，用于在报告里标注出处")

    sub_question: str = Field(
        description=(
            "这条证据是为了回答哪个子问题而收集的 —— 原样抄写分配给你的那个子问题，"
            "不要改写。下游靠它判断「哪些子问题还没搜到足够的证据」。"
        )
    )

    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description=(
            "这条证据的可信程度。依据**信息来源的完整程度**判断，不要凭感觉打分：\n"
            "  high   —— 来自完整读取的网页正文\n"
            "  medium —— 来自搜索结果的摘要片段，未读取正文\n"
            "  low    —— 来源存疑（内容与主题明显无关，或来自低质量聚合站）"
        ),
    )

    citation_index: int | None = Field(
        default=None,
        description=(
            "引用编号 [1][2][3]。**由程序在汇总阶段统一赋值，你不要填写。**"
            "这里一律留空。\n"
            "原因：检索是多个分支**并行**跑的，每个分支都不知道自己是第几个完成，"
            "自己编号必然撞号 —— 而引用编号一旦错乱，整份报告就不可信了。"
        ),
    )


# ---------------------------------------------------------------------------
# 为什么必须有这一层容器：with_structured_output 要求 schema 顶层是对象，
# 不能直接传 list[Evidence]（实测报 TypeError: ... is not a module, class...）。
# 所以「一组证据」必须包成这样一层。这也是类 docstring 不能乱写的原因 ——
# 「不能直接传 list」这种话发给模型毫无意义。
# ---------------------------------------------------------------------------
class EvidenceList(BaseModel):
    """一组证据。"""

    items: list[Evidence] = Field(
        description=(
            "从本轮检索内容中提炼出的全部证据。"
            "如果这些网页确实无法支撑任何结论，返回**空列表** —— "
            "不要为了凑数而产出没有原文依据的证据。"
        ),
        # 【这里刻意不设 max_length —— 曾经设过 12，然后翻车了】
        #
        # 原来写的是 max_length=12，想当「成本护栏」。实际发生的：
        # 模型某次返回 13 条 → Pydantic 抛 ValidationError →
        # RetryPolicy 忠实地重试 3 次 → 每次都拿到 13 条 → 整轮运行终止，
        # 4 个并行分支已经烧掉的检索额度全部白费。
        #
        # 教训：**硬性上界会杀死整轮运行，而它保护的东西根本不值这个代价。**
        # 多几条证据的成本，远远低于一次运行全废的成本。
        #
        # 更要命的是：这是个**确定性失败**，重试救不了。
        # RetryPolicy 只对「偶发失败」有意义 —— 模型这次给 13 条，
        # 下次还会给 13 条，因为问题出在我们给的约束上，不在模型身上。
        #
        # 要控制数量，写在**提示词**里（"最多提炼 N 条"），
        # 而不是写成会抛异常的 schema 约束。
        # 这与 M2 学到的那条一致：schema 只负责拦明显跑偏的情况。
    )


# ============================================================================
# 缺口判定 —— 自我反思的产物
# ============================================================================
class GapVerdict(BaseModel):
    """对「当前证据是否足以回答研究问题」的判断。"""

    sufficient: bool = Field(
        description=(
            "现有证据是否已经足以写出一份完整、有依据的研究报告。\n"
            "判断标准是「能不能回答原始研究问题的所有关键维度」，"
            "而不是「有没有找到一点点资料」。\n"
            "宁可判 insufficient 也不要放过明显的空缺 —— "
            "下一轮补搜比写一份有洞的报告便宜得多。"
        )
    )

    reason: str = Field(
        description="判断的依据，一句话说明为什么够或不够"
    )

    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "具体缺什么。必须是**具体的空缺**，例如「缺少欧洲市场的关税数据」，"
            "而不是「信息不够全面」这种空话。\n"
            "如果 sufficient 为 true，返回空列表。"
        ),
    )

    follow_up_queries: list[str] = Field(
        default_factory=list,
        description=(
            "为填补空缺而生成的补充检索词，1-3 条。\n"
            "每条都必须是能直接拿去搜索引擎检索的完整问句。\n"
            "**不要重复已经搜过的内容** —— 重复检索只是浪费额度。\n"
            "如果 sufficient 为 true，返回空列表。"
        ),
        max_length=3,
    )


if __name__ == "__main__":
    from utils.console import console, rule

    rule("ResearchPlan")
    plan = ResearchPlan(
        rationale="演示用：按「是什么 / 怎么做 / 有什么坑」三层拆解",
        sub_questions=["问题一", "问题二", "问题三"],
    )
    console.print(plan)

    rule("Evidence —— 一条证据长什么样")
    sample = Evidence(
        claim="LangGraph 的检查点系统会在每个超步结束时序列化当前状态并写入持久化后端。",
        quote="检查点系统使得长时间运行的工作流能够从中断点恢复，提供了容错能力和持久化",
        source_url="https://blog.csdn.net/u013172930/article/details/148036704",
        source_title="【LangGraph】什么是检查点（Checkpointer）",
        sub_question="LangGraph 检查点在什么时机被写入？",
        confidence="high",
    )
    console.print(sample)

    rule("Evidence 的 JSON Schema（这就是模型看到的东西）")
    console.print_json(data=Evidence.model_json_schema())

    rule("自查清单（每个字段的 description 都要能回答这几个问题）")
    console.print(
        "  1. [bold]quote[/bold] 有没有说清「找不到原文引用就别产出这条证据」？\n"
        "     [dim]这是防幻觉的物理保证：模型想编结论，但编不出原文引用。[/dim]\n\n"
        "  2. [bold]citation_index[/bold] 有没有显式禁止模型填写？\n"
        "     [dim]模型看到 schema 里有这个字段，天然会想填 —— 必须明确说「不要填」，"
        "并在描述里给出理由。[/dim]\n\n"
        "  3. [bold]confidence[/bold] 有没有给出**客观判断依据**？\n"
        "     [dim]只说「可信程度」的话模型会凭感觉打分，5 条全是 high。"
        "必须给出「什么情况算什么级别」的判据。[/dim]\n\n"
        "  4. [bold]EvidenceList[/bold] 有没有允许返回空列表？\n"
        "     [dim]不允许的话，模型会为了凑数编造证据。这是防幻觉的兜底。[/dim]\n"
    )
