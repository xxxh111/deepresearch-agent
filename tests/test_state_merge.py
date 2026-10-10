"""状态合并与 aggregator 编号的测试。

这些是纯函数级别的测试 —— 不用 LLM、不联网，几毫秒跑完。
"""

import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

from graph.nodes import aggregator_node
from schemas.research import Evidence


class _State(TypedDict, total=False):
    # 字段名必须和 aggregator_node 读写的 key 一致（它用 "evidence"）——
    # 名字对不上不会报错，只是节点的写入会落到一个没人看的通道上，
    # 表现为「调用成功但结果没变」。我第一次写测试就踩了这个。
    evidence: Annotated[list[Evidence], operator.add]
    counter: int


def _ev(url: str, claim: str, index: int | None = None) -> Evidence:
    return Evidence(
        claim=claim,
        quote="原文片段",
        source_url=url,
        source_title=f"来源 {url}",
        sub_question="子问题",
        citation_index=index,
    )


# ============================================================================
# reducer 行为
# ============================================================================
def test_operator_add_merges_parallel_writes():
    """模拟两个并行分支各写一次 items —— 应该合并而不是互相覆盖。"""

    def branch_a(state):
        return {"evidence": [_ev("https://a.com", "A")]}

    def branch_b(state):
        return {"evidence": [_ev("https://b.com", "B")]}

    builder = StateGraph(_State)
    builder.add_node("a", branch_a)
    builder.add_node("b", branch_b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)

    result = builder.compile().invoke({})
    assert len(result["evidence"]) == 2


def test_branch_must_wrap_element_in_list():
    """回归测试：分支返回 `{"evidence": 单条}` 会 TypeError。

    operator.add 执行的是 `list + Evidence`，不是我们想要的追加。
    """

    def bad_branch(state):
        return {"evidence": _ev("https://a.com", "A")}   # ← 忘了包成列表

    builder = StateGraph(_State)
    builder.add_node("bad", bad_branch)
    builder.add_edge(START, "bad")
    builder.add_edge("bad", END)
    graph = builder.compile()

    with pytest.raises(TypeError):
        graph.invoke({})


def test_overwrite_bypasses_reducer():
    """回归测试：有 reducer 的字段想「替换」必须用 Overwrite。

    直接返回列表是**追加** —— 列表会翻倍，而且不报错。
    """

    def replace(state):
        return {"evidence": Overwrite([_ev("https://c.com", "C")])}

    builder = StateGraph(_State)
    builder.add_node("r", replace)
    builder.add_edge(START, "r")
    builder.add_edge("r", END)

    result = builder.compile().invoke({"evidence": [_ev("https://a.com", "A")]})
    assert len(result["evidence"]) == 1, "Overwrite 应该是替换，不是追加"


def test_counter_without_reducer_is_overwritten_not_accumulated():
    """计数器字段（没有 reducer）是「覆盖」语义 —— 这正是我们要的。"""

    def inc(state):
        return {"counter": state.get("counter", 0) + 1}

    builder = StateGraph(_State)
    builder.add_node("i", inc)
    builder.add_edge(START, "i")
    builder.add_edge("i", END)

    result = builder.compile().invoke({})
    assert result["counter"] == 1
    assert isinstance(result["counter"], int), "加 reducer 的话这里会变成 list"


# ============================================================================
# aggregator：去重编号
# ============================================================================
def _run_aggregator(evidence: list[Evidence]) -> list[Evidence]:
    builder = StateGraph(_State)
    builder.add_node("agg", aggregator_node)
    builder.add_edge(START, "agg")
    builder.add_edge("agg", END)

    # aggregator 里会 emit()，在非流式上下文里是空操作
    return builder.compile().invoke({"evidence": evidence})["evidence"]


def test_aggregator_keeps_every_evidence():
    """同 URL 的多条证据都要保留 —— 只是共用同一个引用编号。"""
    result = _run_aggregator([
        _ev("https://a.com", "A1"),
        _ev("https://b.com", "B1"),
        _ev("https://a.com", "A2"),
    ])

    assert len(result) == 3, "条数不能变（用 Overwrite 而不是追加）"
    assert all(e.citation_index for e in result)


def test_aggregator_same_url_shares_citation_index():
    result = _run_aggregator([
        _ev("https://a.com", "A1"),
        _ev("https://b.com", "B1"),
        _ev("https://a.com", "A2"),
    ])
    by_claim = {e.claim: e.citation_index for e in result}

    assert by_claim["A1"] == by_claim["A2"], "同一来源必须共用编号"
    assert by_claim["B1"] != by_claim["A1"]


def test_aggregator_numbers_are_continuous_from_one():
    result = _run_aggregator([
        _ev("https://a.com", "A"),
        _ev("https://b.com", "B"),
        _ev("https://c.com", "C"),
    ])
    indices = sorted({e.citation_index for e in result})
    assert indices == [1, 2, 3]


def test_aggregator_uses_first_appearance_order():
    """★ 回归测试：编号必须按首次出现顺序，不能用 set。

    set 的遍历顺序依赖字符串哈希，而 Python 有进程级哈希随机化 ——
    用 set 的话编号每次运行都不一样，报告不可复现。
    """
    result = _run_aggregator([
        _ev("https://zzz.com", "第一条"),
        _ev("https://aaa.com", "第二条"),
        _ev("https://mmm.com", "第三条"),
    ])
    order = {e.claim: e.citation_index for e in result}

    assert order["第一条"] == 1
    assert order["第二条"] == 2
    assert order["第三条"] == 3


def test_aggregator_handles_empty_evidence():
    assert _run_aggregator([]) == []
