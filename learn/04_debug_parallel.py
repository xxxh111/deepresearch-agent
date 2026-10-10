"""定位 M4 并行崩溃：真实子图 + 真实 LLM，跑 2 个并行分支。

用它把问题从「整张图」缩小到「并行子图」这一个变量上。
"""

import sys
import traceback
from pathlib import Path
from typing import Annotated, TypedDict
import operator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.console  # noqa: E402,F401
from utils.console import console, rule  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Send  # noqa: E402

from graph.subgraphs.researcher import build_researcher  # noqa: E402
from schemas.research import Evidence  # noqa: E402


class PState(TypedDict, total=False):
    topic: str
    questions: list[str]
    evidence: Annotated[list[Evidence], operator.add]


QUESTIONS = [
    "LangGraph 的 checkpointer 和 store 有什么区别",
    "LangGraph 的 interrupt 是怎么实现暂停的",
]


def seed(state: PState) -> dict:
    return {"questions": QUESTIONS}


def fan(state: PState):
    console.print(f"  fan-out：派发 {len(state['questions'])} 个分支")
    return [
        Send("worker", {"topic": state["topic"], "sub_question": q, "round": 1})
        for q in state["questions"]
    ]


def agg(state: PState) -> dict:
    evidence = state.get("evidence") or []
    console.print(f"  aggregator 收到 {len(evidence)} 条证据")
    if evidence:
        console.print(f"  第一条的类型：{type(evidence[0]).__name__}")
        console.print(f"  第一条的字段：{list(evidence[0].model_dump().keys())}")
    return {}


p = StateGraph(PState)
p.add_node("seed", seed)
p.add_node("worker", build_researcher())
p.add_node("agg", agg)
p.add_edge(START, "seed")
p.add_conditional_edges("seed", fan, ["worker"])
p.add_edge("worker", "agg")
p.add_edge("agg", END)
graph = p.compile()

rule("并行跑 2 个真实 researcher 子图")
try:
    for chunk in graph.stream(
        {"topic": "LangGraph 的持久化机制"}, stream_mode=["updates", "custom"]
    ):
        mode, data = chunk
        if mode == "updates":
            for node, update in data.items():
                keys = list(update.keys()) if isinstance(update, dict) else update
                console.print(f"  [{node}] 写回的 key: {keys}")
        else:
            console.print(f"  [custom] {data}")

    rule("结果")
    console.print("[green]没有崩溃[/green]")

except Exception:
    rule("崩溃了")
    console.print("[red]完整 traceback：[/red]")
    traceback.print_exc()
