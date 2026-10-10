"""图结构冒烟测试 —— 不调用 LLM、不联网。

验证的是「接线」：节点都在、循环边还在、条件边的出口没被改坏。
重构之后跑一下这个，能立刻发现图是不是被接错了。
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph

EXPECTED_NODES = {
    "planner",
    "human_review",
    "researcher",
    "aggregator",
    "gap_checker",
    "writer",
    "qa_followup",
}


@pytest.fixture(scope="module")
def graph():
    # 显式传 InMemorySaver，避免测试时碰到 data/checkpoints/research.sqlite
    return build_graph(checkpointer=InMemorySaver())


def test_graph_compiles(graph):
    assert graph is not None


def test_all_nodes_present(graph):
    drawable = graph.get_graph()
    # 注意：drawable.nodes 产出的是**字符串**（节点名），不是对象
    node_ids = set(drawable.nodes)
    assert EXPECTED_NODES <= node_ids, f"缺少节点：{EXPECTED_NODES - node_ids}"


def test_self_reflection_loop_exists(graph):
    """gap_checker → researcher 这条回边是「自我反思」的物理实现。

    删掉它，项目就退化成「搜一次就写报告」的一次性流程。
    """
    drawable = graph.get_graph()
    edges = {(e.source, e.target) for e in drawable.edges}

    assert ("gap_checker", "researcher") in edges, "自我反思的循环边不见了"
    assert ("researcher", "aggregator") in edges
    assert ("aggregator", "gap_checker") in edges


def test_human_in_the_loop_edges(graph):
    """人工审核的两条出边：打回重做 + 通过后并行扇出。"""
    drawable = graph.get_graph()
    edges = {(e.source, e.target) for e in drawable.edges}

    assert ("planner", "human_review") in edges
    assert ("human_review", "planner") in edges
    assert ("human_review", "researcher") in edges


def test_entry_router_has_both_paths(graph):
    """入口必须同时支持「新研究」和「追问」。"""
    drawable = graph.get_graph()
    edges = {(e.source, e.target) for e in drawable.edges}

    assert ("__start__", "planner") in edges
    assert ("__start__", "qa_followup") in edges


def test_subgraph_has_no_checkpointer():
    """回归测试：子图绝不能自带 checkpointer。

    带上之后被父图嵌入时会直接报错。这条是从网上抄代码最常见的坑。
    """
    from graph.subgraphs.researcher import build_researcher

    sub = build_researcher()
    # 自带 checkpointer 的图，checkpointer 属性非 None
    assert getattr(sub, "checkpointer", None) is None


def test_subgraph_declares_output_schema():
    """回归测试：子图必须声明 output_schema。

    不声明的话，子图会把整个最终状态交回父图 —— 包括 topic 这种入参。
    N 个并行分支各写一次 topic，而父图没有它的 reducer，
    直接 InvalidUpdateError（这是实际踩过的坑）。
    """
    from graph.state import ResearcherOutput, ResearcherState
    from graph.subgraphs.researcher import build_researcher

    sub = build_researcher()
    assert sub.stream_channels is not None

    # 出参 schema 里只应该有 evidence
    assert set(ResearcherOutput.__annotations__) == {"evidence"}
    # 而入参状态里有更多字段（topic / sub_question / messages 等）
    assert len(ResearcherState.__annotations__) > len(ResearcherOutput.__annotations__)
