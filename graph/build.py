"""图装配。

所有节点和边集中在这一个文件里接线 —— 排查「流程为什么这样走」时只看这里。

LangGraph 建图四步（M1 的全部内容）：
    1. builder = StateGraph(OverAllState)   声明状态类型
    2. add_node(...)                        注册节点
    3. add_edge / add_conditional_edges     连接节点
    4. builder.compile()                    编译成可执行图

后续里程碑会持续改这个文件：
    M4 加 Send 条件边（并行扇出）    M5 加 checkpointer 参数
    M6 加 interrupt 人工审核节点     M7 加缺口检查循环 + 入口路由

注意本文件只负责**装配**：定义策略（比如重试规则）不在这里，
在 utils/retry.py，这里只是 import 进来用。
"""

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    aggregator_node,
    gap_checker_node,
    human_review_node,
    planner_node,
    qa_followup_node,
    writer_node,
)
from graph.routers import route_after_gap, route_after_review, route_entry
from graph.state import OverAllState
from graph.subgraphs.researcher import build_researcher
from utils.checkpointer import get_checkpointer
from utils.retry import llm_retry_policy


def build_graph(checkpointer=None):
    """构建并编译主图。

    Args:
        checkpointer: 持久化后端。不传则按 config/research.yml 的
            `checkpointer` 配置自动选择（memory | sqlite）。
            显式传 None 也一样走自动选择 —— 想看「不持久化」的行为，
            把配置改成 memory 即可（进程内有效，退出即忘）。

    Returns:
        编译好的图（CompiledStateGraph），可以直接 .invoke() / .stream()。
    """
    if checkpointer is None:
        checkpointer = get_checkpointer()

    builder = StateGraph(OverAllState)

    # ---------- 节点 ----------
    # retry_policy 来自 utils/retry.py。
    # 注意它**不是**裸的 RetryPolicy(max_attempts=3) ——
    # 那样写的话，模型输出不合 schema 时抛出的 ValidationError 会因为
    # 继承自 ValueError 而被判定为「代码 bug，不重试」，重试保护等于没有。
    # 详见 utils/retry.py 顶部的说明。
    builder.add_node("planner", planner_node, retry_policy=llm_retry_policy())

    # 【M3】把 researcher 子图作为一个节点挂进父图。
    #
    # 注意这里传的是**编译好的子图**（build_researcher() 的返回值），
    # 而不是一个函数。LangGraph 能识别它是子图，并自动处理状态进出：
    #   · 进入：按**同名匹配**把父图状态里的 topic / sub_question / round 交给子图
    #   · 出来：子图的 evidence 按**父图的 reducer 规则**合并回父图
    #           （子图里没有对应父图 key 的字段会被静默丢弃）
    #
    # 子图内部**不能**自带 checkpointer —— 持久化统一由父图负责，否则会报错。
    builder.add_node("researcher", build_researcher())

    # 【M6】人工审核节点。
    # 它内部调用 interrupt()，图会在这里暂停并把状态写进检查点。
    #
    # 注意它**没有** retry_policy —— 这个节点不调用模型，
    # 而且重试一个「等人做决定」的操作毫无意义。
    builder.add_node("human_review", human_review_node)

    # 汇总节点：单写者、确定性。并行分支做不到的「全局去重 + 编号」在这里做。
    builder.add_node("aggregator", aggregator_node)

    # 【M7】自我反思 + 产出
    builder.add_node("gap_checker", gap_checker_node, retry_policy=llm_retry_policy())
    builder.add_node("writer", writer_node, retry_policy=llm_retry_policy())
    builder.add_node("qa_followup", qa_followup_node, retry_policy=llm_retry_policy())

    # ---------- 边 ----------
    #
    # 完整流程图：
    #
    #        START ┄> planner ──> human_review ┄> researcher ──> aggregator
    #             ┄> qa_followup                              ↑        │
    #                        ┌────────────────────────────────┘        ↓
    #                        └──────────────── gap_checker ┄> writer ──> END
    #
    #        human_review 的出边：打回 → planner / 通过 → researcher
    #        gap_checker  的出边：补搜 → researcher（循环回上游）/ 够了 → writer

    # 入口路由：新研究走 planner，追问直接走 qa_followup（跳过全部检索）
    builder.add_conditional_edges(START, route_entry, ["planner", "qa_followup"])

    builder.add_edge("planner", "human_review")

    # 条件边可以返回节点名（打回）或 Send 列表（并行扇出），见 routers.py
    builder.add_conditional_edges(
        "human_review", route_after_review, ["planner", "researcher"]
    )

    # 所有并行分支执行完毕、状态合并之后，才会执行 aggregator
    # —— 这就是「超步」语义：同一超步内的分支全部完成后，才进入下一个超步。
    builder.add_edge("researcher", "aggregator")

    # ★ M7 的循环：aggregator → gap_checker →（补搜）→ researcher → aggregator
    #
    # 这是个**有界**循环 —— 出口由 route_after_gap 控制，
    # 三重保险：max_iterations（业务上限）+ recursion_limit（LangGraph 硬顶）
    # + follow_up_queries 为空时强制退出。
    builder.add_edge("aggregator", "gap_checker")
    builder.add_conditional_edges(
        "gap_checker", route_after_gap, ["researcher", "writer"]
    )

    builder.add_edge("writer", END)
    builder.add_edge("qa_followup", END)

    return builder.compile(checkpointer=checkpointer)


# 想看图的 Mermaid 结构：python scripts/show_graph.py
#
# 这里刻意不写 __main__ 块 —— graph/__init__.py 已经 re-export 了 build_graph，
# 所以 import graph 时会先加载本模块；再用 python -m graph.build 跑它，
# runpy 会重复执行一遍并抛 RuntimeWarning。放到 scripts/ 下就没这个问题。
