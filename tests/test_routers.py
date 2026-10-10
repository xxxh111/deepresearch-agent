"""路由函数测试。

条件边是全图的控制流，也是最容易出静默错误的地方 ——
所以这里逐个出口都测一遍。
"""

import pytest
from langgraph.types import Send

from graph.routers import dispatch_all, route_after_gap, route_after_review, route_entry
from schemas.research import GapVerdict, ResearchPlan


def _plan(*questions: str) -> ResearchPlan:
    """构造一个合法的 ResearchPlan。

    注意 ResearchPlan 的 sub_questions 有 min_length=3 约束 ——
    这就是 M2 讲的「schema 是护栏」，测试里传 1 条会直接 ValidationError。
    这里自动补齐到 3 条，让测试专注于被测的逻辑本身。
    """
    items = list(questions)
    while len(items) < 3:
        items.append(f"占位问题 {len(items) + 1}")
    return ResearchPlan(rationale="测试用", sub_questions=items)


def _verdict(sufficient: bool, queries: list[str] | None = None) -> GapVerdict:
    return GapVerdict(
        sufficient=sufficient,
        reason="测试用",
        gaps=[] if sufficient else ["缺少某个维度的数据"],
        follow_up_queries=queries or [],
    )


# ============================================================================
# route_entry
# ============================================================================
def test_route_entry_new_research():
    assert route_entry({"topic": "某主题"}) == "planner"


def test_route_entry_followup():
    assert route_entry({"followup_question": "某个追问"}) == "qa_followup"


# ============================================================================
# dispatch_all —— 并行扇出
# ============================================================================
def test_dispatch_creates_one_send_per_question():
    state = {"topic": "T", "plan": _plan("q1", "q2", "q3")}
    sends = dispatch_all(state)

    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)
    assert [s.node for s in sends] == ["researcher"] * 3


def test_dispatch_payload_carries_all_subgraph_inputs():
    """Send 的 payload 是分支的**完整**输入状态，不是和父图合并。

    所以 topic / sub_question / round 必须全部带上 ——
    漏掉任何一个，子图里对应的字段就是空的（而且不报错）。
    """
    state = {"topic": "研究主题", "plan": _plan("q1")}
    send = dispatch_all(state)[0]

    assert send.arg["topic"] == "研究主题"
    assert send.arg["sub_question"] == "q1"
    assert send.arg["round"] == 1


def test_dispatch_rejects_empty_plan():
    with pytest.raises(ValueError):
        dispatch_all({"topic": "T", "plan": None})


# ============================================================================
# route_after_review
# ============================================================================
def test_review_rejected_goes_back_to_planner():
    state = {"approved": False, "plan": _plan("q1")}
    assert route_after_review(state) == "planner"


def test_review_approved_fans_out():
    plan = _plan("q1", "q2", "q3", "q4")
    state = {"approved": True, "topic": "T", "plan": plan}
    sends = route_after_review(state)
    assert isinstance(sends, list) and len(sends) == 4


# ============================================================================
# route_after_gap —— 循环闸门（四条出口）
# ============================================================================
def test_gap_sufficient_goes_to_writer():
    state = {"iteration": 1, "gap_verdict": _verdict(True)}
    assert route_after_gap(state) == "writer"


def test_gap_insufficient_fans_out_followup():
    """唯一的循环出口：判了不足、还有轮次、且给出了可搜的问句。"""
    state = {
        "topic": "T",
        "iteration": 1,
        "gap_verdict": _verdict(False, ["补充问题 A", "补充问题 B"]),
    }
    sends = route_after_gap(state)

    assert isinstance(sends, list) and len(sends) == 2
    assert sends[0].arg["sub_question"] == "补充问题 A"
    assert sends[0].arg["round"] == 2, "补搜的轮次应该是 iteration + 1"


def test_gap_at_iteration_limit_goes_to_writer():
    """到迭代上限时即使判定不足也要去 writer —— 诚实降级。"""
    state = {
        "topic": "T",
        "iteration": 99,          # 远超配置的 max_iterations
        "gap_verdict": _verdict(False, ["还想再搜"]),
    }
    assert route_after_gap(state) == "writer"


def test_gap_insufficient_without_queries_goes_to_writer():
    """回归测试：返回空 Send 列表会让图**静默终止**。

    这个出口就是为了拦住那种情况 —— 判了不足但没给出可执行的检索词时，
    必须直接去 writer，不能返回 []。
    """
    state = {"topic": "T", "iteration": 1, "gap_verdict": _verdict(False, [])}
    result = route_after_gap(state)

    assert result == "writer"
    assert not isinstance(result, list), "绝不能返回空列表 —— 那会让图静默终止"


def test_gap_without_verdict_goes_to_writer():
    assert route_after_gap({"iteration": 1, "gap_verdict": None}) == "writer"
