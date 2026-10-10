"""图状态定义。

============================ 这是整个项目最关键的文件 ============================

并行分支的合并规则全部在这里，M4（Send 并行扇出）的成与败取决于它。
下面三条铁律，每一条都是踩坑踩出来的 —— 面试也最爱问这些：

【铁律一】会被多个并行分支写入的 key，**必须**有 reducer
    多个分支在同一超步里同时写同一个 key 时，LangGraph 无法自行决定保留谁，
    于是抛 InvalidUpdateError。必须由你提供合并函数来消除歧义。
    写法：  evidence: Annotated[list[Evidence], operator.add]

【铁律二】用 operator.add 时，分支返回的**必须是列表**
    写  {"evidence": [e]}    ✅   （外面包一层 list）
    写  {"evidence": e}      ❌   operator.add 会执行 list + Evidence → TypeError

【铁律三】计数器**不能**用 operator.add
    iteration 这类字段如果用 add，会把每轮的值累加成一个 list（[1]），
    于是所有 `iteration < max_iterations` 判断全部失效，循环直接失控。
    计数器是「单写者」字段，让 LangGraph 的默认覆盖语义处理即可 ——
    也就是：什么都不用写，不加 Annotated 就行。

另外一条经验（M4 会验证）：
    reducer 的合并**顺序不可保证**。所以任何需要稳定顺序的逻辑
    （比如给引用编号 [1][2][3]）都**不能**在并行分支里做，
    必须放到下游那个确定性的 aggregator 节点里做。

=================================================================================
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from schemas.research import Evidence, GapVerdict, ResearchPlan

# --- M4 及以后会用到的字段，这里先留出形状说明 -------------------------------
#
#     evidence: Annotated[list[Evidence], operator.add]   # 铁律一 + 铁律二
#     iteration: int                                      # 铁律三：不加 reducer
#     messages: Annotated[list[AnyMessage], add_messages] # 多轮追问
#
# 现在（M1）只需要跑通「接线」，所以状态保持最小。


class OverAllState(TypedDict, total=False):
    """主图状态。

    total=False 表示所有字段都是可选的 —— 节点返回部分更新即可，
    不需要把整个 state 再写回来（这是 LangGraph 的节点契约）。
    """

    # 用户输入的原始研究主题
    topic: str

    # 规划节点产出的研究计划（M2 起由 LLM 生成）
    plan: ResearchPlan | None

    # ---------- 人工审核（M6）----------
    approved: bool             # 用户是否通过了研究计划
    feedback: str              # 被打回时用户给的意见，planner 会把它带进下一轮提示词
    # 这两个字段都是「单写者」—— 只有 human_review 节点写它们，
    # 所以不需要 reducer。

    # ---------- 传给 researcher 子图的字段 ----------
    # 子图通过「同名匹配」从父图状态里取输入，所以这里必须有这两个 key，
    # 否则子图拿到的 sub_question 是空的。
    sub_question: str          # 当前正在研究的那个子问题
    round: int                 # 第几轮检索

    # ---------- 从 researcher 子图收回的字段 ----------
    # ⚠️ key 必须与 ResearcherState 里的出参**完全同名**，
    #    否则子图的产出会被静默丢弃（不报错，你只会发现证据是空的）。
    #
    # 【M4：这里加上了 reducer，因为开始并行了】
    # 现在有 N 个 researcher 分支在**同一个超步内并发执行**，各自往 evidence 里写。
    # 没有 reducer 的话 LangGraph 无法决定保留谁，直接抛 InvalidUpdateError。
    #
    # 用了 operator.add 之后，分支返回值**必须包成列表**：
    #     return {"evidence": [e]}    ✅  operator.add 收到 list
    #     return {"evidence": e}      ❌  list + Evidence → TypeError
    #
    # ⚠️ 合并顺序**不可保证**！N 个分支谁先跑完是不确定的。
    #    所以任何需要稳定顺序的逻辑（比如引用编号 [1][2][3]）
    #    都不能在这里做，必须放到下游的 aggregator 节点（单写者、确定性）。
    evidence: Annotated[list[Evidence], operator.add]

    # ---------- 自我反思与循环（M7）----------
    # 已完成的研究轮次。**单写者**（只有 aggregator 写它），所以不加 reducer。
    #
    # ⚠️ 铁律三的实例：这个字段绝对不能加 operator.add。
    #    加了的话每经过一个节点就累加一次，读出来是个 list，
    #    所有 `iteration < max_iterations` 判断全部失效 —— 循环直接失控。
    iteration: int

    gap_verdict: GapVerdict | None    # 缺口判定结果（结构化输出）

    # ---------- 产出 ----------
    report: str          # 最终 Markdown 报告全文
    report_path: str     # 落盘位置，方便 CLI 告诉用户「去哪儿看」

    # ---------- 多轮追问（M7）----------
    # 入口路由靠它判断这次是「新研究」还是「追问」：
    # 有 followup_question 就走 qa_followup，否则走 planner。
    followup_question: str

    # 追问的对话历史。用 add_messages —— 多轮追问要一轮轮累积。
    messages: Annotated[list[AnyMessage], add_messages]


# ============================================================================
# 子图私有状态
# ============================================================================
# 为什么子图需要自己的状态类型？
#
# researcher 子图内部要维护一长串「LLM 决定搜什么 → 工具返回了什么」的对话历史。
# 这些东西：
#   · 只对研究员自己有意义 —— 父图拿到它们毫无用处
#   · 体量可能很大（几个网页的正文都躺在里面）—— 塞进父图状态会让
#     M5 的 checkpoint 变得臃肿
#
# 所以给子图一套独立的状态 schema。父图看不见 messages 和 tool_rounds。
#
# ⚠️ 但有一条铁律：**出参的 key 必须与父图同名**。
#    子图返回的字段里，只有父图状态里也存在同名字段的那些才会被合并回去，
#    其余会被**静默丢弃**（不报错、不警告）。
#    所以下面的 evidence 必须和 OverAllState 里的 evidence 拼写一致。
# ============================================================================
class ResearcherState(TypedDict, total=False):
    """researcher 子图的私有状态。"""

    # ---------- 入参：由父图传入 ----------
    topic: str
    sub_question: str          # 本分支负责的那一个子问题
    round: int                 # 第几轮检索（1 = 首轮，>1 = 缺口补充）

    # ---------- 内部工作区：父图看不见 ----------
    messages: Annotated[list[AnyMessage], add_messages]
    # ↑ 必须有 add_messages reducer。
    #   agent 节点和 tools 节点会轮流往这个列表里追加消息，
    #   没有 reducer 的话后写的会覆盖先写的，对话历史就断了。
    #   （对比：OverAllState 里没有 reducer 的字段是「覆盖」语义）

    tool_rounds: int           # 已调用工具的轮次，用于给 ReAct 循环上闸

    # ---------- 出参：key 必须与父图同名 ----------
    evidence: list[Evidence]
    # ↑ 这里是普通 list，不是 Annotated[..., operator.add]。
    #   因为子图内部只有一个 summarize 节点写它，不存在并行写入。
    #   合并到父图时，用的是**父图**的 reducer 规则（父图那边有 operator.add）。


# ============================================================================
# 子图的「出参过滤器」—— M4 踩坑踩出来的必需品
# ============================================================================
# 【发生了什么事】
# 加上 Send 并行扇出之后，图直接崩了：
#
#     InvalidUpdateError: At key 'topic': Can receive only one value per step.
#     Use an Annotated key to handle multiple values.
#
# 原因：**子图会把它的整个最终状态交回父图**，包括那些你只打算当「入参」的字段。
# researcher 子图的最终状态里有 topic（它是入参），于是 N 个并行分支
# 每个都往父图的 topic 写了一次 —— 而 topic 没有 reducer，撞车。
#
# 【为什么这个坑特别难发现】
# 分支并没有「故意」写 topic。是它回显输入时被动带出来的。
# 你盯着 routers.py 和 nodes.py 找半天，也看不出谁在写 topic。
#
# 【为什么用 output_schema 而不是给 topic 加 reducer】
# 给 topic 加 reducer 是错的解法 —— 它掩盖了真正的问题：
# 父图根本不希望子图改写 topic。加 reducer 只是让并发的重复写入「不报错」，
# 语义上仍然是错的（凭什么是最后一个分支的 topic 生效？）。
#
# 正确的做法是**明确声明子图只输出什么**。
# 这正是你在 06_dynamic_send.ipynb 里学过的 input_schema / output_schema ——
# 当时只是知道有这个东西，现在它解决了一个真实问题。
# ============================================================================
class ResearcherOutput(TypedDict, total=False):
    """researcher 子图对外暴露的输出。

    只有这里声明了的 key 才会被合并回父图，其余一律丢弃。
    这既解决了并行冲突，也顺带解决了另一个问题：
    子图内部的 messages（可能几万字）不会再被交回父图污染状态。
    """

    evidence: list[Evidence]
