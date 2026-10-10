"""条件边路由函数。

LangGraph 的条件边函数可以返回**两种**东西，这是它最容易被忽略的能力：

    返回一个节点名字符串
        → 普通的条件分支，只能走一条边

    返回一个 Send 列表                    ← 本文件的主角
        → **动态创建 N 个并行分支**，分支数量在运行时才确定

第二种就是「map-reduce」里的 map。你 06_dynamic_send.ipynb 里想做的
就是这件事 —— 现在它作为生产代码的一部分真正跑起来了。
"""

from langgraph.types import Send

from graph.state import OverAllState
from utils.config_handler import research_conf
from utils.console import emit


def route_entry(state: OverAllState) -> str:
    """入口路由：这次是「新研究」还是「追问」。

    判断依据是状态里有没有 followup_question —— CLI 用 --ask 传进来。
    这个路由让「追问」完全跳过 planner / 检索，直接走向 qa_followup，
    零检索额度。
    """
    if state.get("followup_question"):
        return "qa_followup"
    return "planner"


def route_after_gap(state: OverAllState):
    """缺口检查之后：继续补搜，还是开始写报告。

    ============================ 这是全图的循环闸门 ============================

    有四条可能的出口，三条都通向 writer。**宁可少搜、不可失控**：

        ① 证据充分                    → 写报告（正常出口）
        ② 已到迭代上限                → 写报告，但在报告里明写「仍有缺口」
        ③ 判了不足但没给出可搜的问句   → 写报告（否则 Send 空列表，图会卡住）
        ④ 判了不足、还有轮次、有问句   → **并行补搜**（唯一会循环的出口）

    ⚠️ 出口③ 特别重要：如果 verdict.follow_up_queries 是空列表，
       返回 [] 会让 langgraph 直接报错或原地不动。**任何返回 Send 列表的
       路由都必须考虑「列表为空」这种情况。**

    ===========================================================================
    """
    verdict = state.get("gap_verdict")
    iteration = state.get("iteration", 1)
    max_iterations = int(research_conf.get("max_iterations", 2))

    # ① 没有判定结果（理论上不该发生）或判定充分
    if verdict is None or verdict.sufficient:
        return "writer"

    # ② 到顶了。**诚实降级**：继续写报告，但让 writer 在末尾声明缺口。
    #    这比「不写报告」和「假装没缺口」都好。
    if iteration >= max_iterations:
        emit(f"已达迭代上限 {max_iterations} 轮，带着未解决的缺口进入撰写")
        return "writer"

    # ③ 判了不足，但没能给出可执行的补充检索词
    if not verdict.follow_up_queries:
        emit("判定证据不足，但没有可执行的补充检索词，直接进入撰写")
        return "writer"

    # ④ 并行补搜。Send 的数量由 LLM 决定（1-3 条，schema 里限了上限）
    emit(f"开始第 {iteration + 1} 轮补搜：{len(verdict.follow_up_queries)} 个问题")
    return [
        Send(
            "researcher",
            {
                "topic": state.get("topic", ""),
                "sub_question": question,
                "round": iteration + 1,
            },
        )
        for question in verdict.follow_up_queries
    ]


def route_after_review(state: OverAllState):
    """人工审核之后往哪走。

    这个函数同时返回了两种东西 —— 这正是条件边函数最容易被人忽略的能力：
        · 一个节点名字符串  → 打回重做，走普通分支
        · 一个 Send 列表    → 并行扇出

    两个分支的语义完全不同，但都由这一个函数决定，
    因为「审核通过没有」这个判断只做一次。
    """
    if not state.get("approved", False):
        # 用户打回了计划 → 回到 planner 重新规划。
        # planner 会读 state["feedback"]，把用户意见带进提示词。
        return "planner"

    return dispatch_all(state)


def dispatch_all(state: OverAllState) -> list[Send]:
    """把研究计划里的每个子问题，各派给一个 researcher 分支。

    一个子问题 = 一个 Send = 一个独立的分支。
    这些分支在**同一个超步（superstep）内并发执行**，互相看不见对方的中间结果。

    Send 的第二个参数是这个分支的**输入状态** ——
    它会成为那个 researcher 子图的初始 state。

    注意这里必须把子图需要的输入字段全部带上（topic / sub_question / round）：
    分支拿到的是「超步开始时的状态快照 + 这里给的 payload」，
    不是父图的实时状态。
    """
    plan = state.get("plan")
    if plan is None or not plan.sub_questions:
        raise ValueError("状态里没有研究计划 —— planner 节点必须先跑")

    topic = state.get("topic", "")

    branches = [
        Send("researcher", {"topic": topic, "sub_question": question, "round": 1})
        for question in plan.sub_questions
    ]

    return branches
