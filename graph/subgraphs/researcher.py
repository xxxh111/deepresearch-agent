"""researcher 子图：针对一个子问题检索并提炼证据。

图结构：

    START → prepare → agent ⇄ tools → summarize → END
                        │                 │
                        │  (有 tool_calls) │
                        └─────────────────┘
                        │
                        (没有 tool_calls)

这张图就是你在 Agent项目 里用 `create_agent(model, tools=[...])` 一行拿到的那个循环 ——
只不过那行代码把这张图藏在内部了。这里把它拆开手搭一遍。

每个节点做什么：
    prepare   —— 把 system prompt 和子问题组装成初始消息
    agent     —— 让 LLM 决定「还要不要搜」。要搜就返回带 tool_calls 的消息
    tools     —— ToolNode 自动执行 tool_calls，把结果包成 ToolMessage 塞回消息列表
    summarize —— 读完整段检索对话，提炼成结构化证据

【为什么不直接使用内置的 tools_condition？】
    tools_condition 只做一件事：判断最后一条消息有没有 tool_calls。
    我们还需要一个「最多搜几轮」的闸门来控制成本 —— 所以自己写路由。
    下面是自定义路由，逻辑和 tools_condition 一样，只是多了轮次控制。
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from graph.state import ResearcherOutput, ResearcherState
from model.factory import get_chat_model
from schemas.research import EvidenceList
from tools.search_tools import ALL_TOOLS
from utils.console import emit
from utils.prompt_loader import load_prompt
from utils.retry import llm_retry_policy

# ReAct 循环最多执行几轮工具调用。
# 这是**成本闸门**：每轮工具调用都是一次真实的搜索或网页抽取，都要花额度。
# 调大能搜得更充分，但单篇报告的成本和时间线性上升。
_MAX_TOOL_ROUNDS = 4


# ============================================================================
# 惰性单例
# ============================================================================
# 和 model/factory.py 一样：不在模块顶层构造，保证 import 本文件永远安全
# （缺 API key 时 check_env、单元测试、show_graph 都还能跑）。
@lru_cache(maxsize=1)
def _bound_model():
    """绑定了工具的模型 —— 给 agent 节点用来发起工具调用。"""
    return get_chat_model("researcher").bind_tools(ALL_TOOLS)


@lru_cache(maxsize=1)
def _plain_model():
    """没绑工具的模型 —— 轮次用尽时用它，让模型只能给文字结论。"""
    return get_chat_model("researcher")


@lru_cache(maxsize=1)
def _tool_node() -> ToolNode:
    """ToolNode 只是把函数包装成可执行节点，不碰网络，构造很轻。"""
    return ToolNode(ALL_TOOLS)


@lru_cache(maxsize=1)
def _summarize_chain():
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("summarize_evidence")),
                (
                    "human",
                    "研究主题：{topic}\n"
                    "你负责的子问题：{sub_question}\n"
                    "这是第 {round} 轮检索。\n\n"
                    "以下是完整的研究员检索过程：",
                ),
                # MessagesPlaceholder 会把整个对话历史插进来 ——
                # 这是 summarize 能看到网页内容的唯一途径。
                MessagesPlaceholder("messages"),
            ]
        )
        | get_chat_model("summarize").with_structured_output(EvidenceList)
    )


# ============================================================================
# 节点
# ============================================================================
def prepare_node(state: ResearcherState) -> dict:
    """组装初始消息。"""
    sub_question = state.get("sub_question") or state.get("topic", "")
    topic = state.get("topic", "")

    emit(f"开始检索：{sub_question}")

    return {
        "messages": [
            SystemMessage(content=load_prompt("researcher_search")),
            HumanMessage(
                content=(
                    f"研究主题：{topic}\n"
                    f"你负责的子问题：{sub_question}\n"
                    f"这是第 {state.get('round', 1)} 轮检索，请开始。"
                )
            ),
        ],
        # 显式初始化计数器，免得后面到处写 state.get(..., 0)
        "tool_rounds": 0,
    }


def agent_node(state: ResearcherState) -> dict:
    """让 LLM 决定下一步：继续调用工具，还是给出结论。"""
    rounds = state.get("tool_rounds", 0)

    if rounds >= _MAX_TOOL_ROUNDS:
        # 轮次用尽 → 用**不绑工具**的模型。
        #
        # 为什么不直接在路由里判断？
        #   如果路由在「最后一条消息带 tool_calls」时改道去 summarize，
        #   消息历史里就会留下「有 tool_calls 却没有对应工具结果」的非法序列，
        #   调用模型 API 会直接报错。
        #   而不绑工具的话，模型**根本发不出** tool_calls，
        #   路由会沿着正常路径走向 summarize，历史始终合法。
        #
        # 这是个有用的经验：**在能选择的地方做约束，比在下游亡羊补牢好。**
        emit(f"  已达 {_MAX_TOOL_ROUNDS} 轮工具调用上限，开始收尾")
        response = _plain_model().invoke(state["messages"])
    else:
        response = _bound_model().invoke(state["messages"])

    return {"messages": [response]}


def tools_node(state: ResearcherState) -> dict:
    """执行工具调用，并给轮次计数器 +1。

    ToolNode 会自己把每个 tool_call 执行掉、包成 ToolMessage 追加到消息列表。
    我们只是额外把计数器推进一步。
    """
    result = _tool_node().invoke(state)
    return {**result, "tool_rounds": state.get("tool_rounds", 0) + 1}



def summarize_node(state: ResearcherState) -> dict:
    """把整段检索对话提炼成结构化证据。

    注意返回值的 key 是 `evidence` ——
    必须和父图 OverAllState 里的字段同名，否则会被静默丢弃。
    """
    emit("正在提炼证据…")

    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("summarize_evidence")),
                ("human", 
                 "研究主题：{topic}\n"
                 "你负责的子问题：{sub_question}\n"
                 "这是第{round}轮检索。\n"
                 "以下是完整的研究员检索过程：",
                 ),
                 MessagesPlaceholder("messages")
            ]
        )
    llm = get_chat_model("summarize").with_structured_output(EvidenceList)
    chain = prompt | llm
    result = chain.invoke({
        "topic": state.get("topic", ""),
        "sub_question": state.get("sub_question", ""),
        "round": state.get("round", 1),
        "messages": state["messages"],
    })
    evidence = list(result.items)
    emit(f"提炼出 {len(evidence)} 条证据")
    return {"evidence": evidence}

# ============================================================================
# 路由
# ============================================================================
def route_after_agent(state: ResearcherState) -> str:
    """最后一条消息带 tool_calls → 去执行工具；否则 → 去提炼证据。

    逻辑等同于内置的 langgraph.prebuilt.tools_condition，
    「最多几轮」的闸门由 agent_node 换模型来实现（见那里的说明）。
    """
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "summarize"


# ============================================================================
# 装配
# ============================================================================
def build_researcher():
    """构建 researcher 子图。

    注意两点：

    1. **绝不传 checkpointer**。子图被父图嵌入时自带 checkpointer 会直接报错 ——
       持久化统一由父图负责，子图只提供结构。

    2. **必须传 output_schema**。不给的话子图会把整个最终状态交回父图，
       包括 topic 这类只打算当输入的字段 —— 而 N 个并行分支各写一次 topic
       就会撞上 InvalidUpdateError（父图里 topic 没有 reducer）。
       详见 graph/state.py 里 ResearcherOutput 的说明。
    """
    builder = StateGraph(ResearcherState, output_schema=ResearcherOutput)

    builder.add_node("prepare", prepare_node)
    builder.add_node("agent", agent_node, retry_policy=llm_retry_policy())
    builder.add_node("tools", tools_node)
    builder.add_node("summarize", summarize_node, retry_policy=llm_retry_policy())

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "agent")

    # 条件边：agent 之后往哪走由 route_after_agent 决定
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "summarize": "summarize"},
    )

    # ★ 这条边构成了**循环** —— agent ⇄ tools 可以来回走多轮。
    #   这是 M3 和前面所有图最本质的区别：之前都是单向的直线。
    #   循环也意味着可能失控，所以要靠 _MAX_TOOL_ROUNDS 上闸。
    builder.add_edge("tools", "agent")

    builder.add_edge("summarize", END)

    return builder.compile()


# ============================================================================
# 自测：不经过父图，单独跑一个子问题
# ============================================================================
if __name__ == "__main__":
    from utils.console import console, rule

    graph = build_researcher()

    rule("子图结构")
    console.print(graph.get_graph().draw_mermaid())

    rule("单独运行：一个子问题")
    console.print("[dim]这同时验证了子图可以独立测试 —— 不必拉起整张父图[/dim]\n")

    result = graph.invoke(
        {
            "topic": "LangGraph 的检查点机制",
            "sub_question": "LangGraph 检查点的存储结构和序列化格式是怎样的",
            "round": 1,
        }
    )

    evidence = result.get("evidence", [])
    rule(f"得到 {len(evidence)} 条证据")
    for index, item in enumerate(evidence, 1):
        console.print(f"\n[bold cyan]{index}.[/bold cyan] {item.claim}")
        console.print(f"   [dim]引用：{item.quote[:90]}…[/dim]")
        console.print(f"   [dim]来源：{item.source_title[:50]}[/dim]")
        console.print(f"   [dim]可信度：{item.confidence}[/dim]")
