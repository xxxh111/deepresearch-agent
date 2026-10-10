"""Streamlit 界面。

===============================================================================
理解这个文件的关键：Streamlit 的执行模型
===============================================================================

Streamlit **每次交互都会把整个脚本从头重跑一遍**（点按钮、输入文字、刷新……）。
局部变量全部丢失，只有 `st.session_state` 能跨次存活。

但有两条硬性规则，我第一版就是踩在这上面：

  规则一：**每个 widget 的 key 必须唯一**，否则
          StreamlitDuplicateElementKey

  规则二：**widget 一旦实例化，它的 session_state key 就不能再改**。
          所以「按钮点击后要设置 thread_id」这种需求，
          不能写成「if 按钮被点: st.session_state.thread_id = ...」——
          因为此时 sidebar 里的输入框早就渲染过了。

解法：**用回调（on_click）**。回调在脚本重跑**之前**执行，那时可以随意改
session_state（包括 widget 的 key）。所以本文件里所有「会改变状态」的按钮
都是 `st.button(..., on_click=_cb_xxx)`，而不是 `if st.button(...):`。

===============================================================================
阶段机
===============================================================================

    idle      等待用户输入研究主题
    running   正在跑图（流式显示进度）
    review    ⏸ 图停在 interrupt 上，等用户审核计划
    done      报告已完成，可以追问

    idle --[开始研究]--> running --[遇到中断]--> review --[提交]--> running
                              └────[跑完]────> done --[追问]--> running
"""

import uuid

import streamlit as st
from langgraph.types import Command

import utils.console  # noqa: F401  —— 副作用：修复编码
from utils.config_handler import research_conf
from utils.console import sanitize_text
from utils.tracing import setup_tracing


# ============================================================================
# 图的单例
# ============================================================================
@st.cache_resource(show_spinner=False)
def _get_graph():
    """整个 Streamlit 进程共用一个编译好的图。

    真正的关键是**检查点后端**：utils/checkpointer.py 里做了模块级单例，
    所以多次重跑脚本拿到的是同一个 saver —— 用户刷新页面后研究历史还在。
    """
    from graph import build_graph

    return build_graph()


# ============================================================================
# session_state
# ============================================================================
_DEFAULTS = {
    "phase": "idle",
    "thread_id": "",       # ⚠️ 这个 key 同时是 sidebar 输入框的 key
    "payload": None,       # 下一次推进图时用的输入（dict 或 Command）
    "pending": None,       # 待用户处理的中断
    "progress": [],        # 进度日志
    "report": "",          # 报告正文（流式累积）
    "answer": "",          # 追问的回答
    "topic": "",           # 当前研究主题
    "flash": "",           # 一次性的提示消息
}


def _init_state() -> None:
    for key, value in _DEFAULTS.items():
        st.session_state.setdefault(key, value)


# ============================================================================
# 回调 —— 全部在脚本重跑之前执行，所以可以安全地改 session_state
# ============================================================================
def _cb_start() -> None:
    topic = (st.session_state.get("topic_input") or "").strip()
    if not topic:
        st.session_state.flash = "请先填写研究主题"
        return

    # 只有在回调里才能这样写 —— 此时 sidebar 的输入框还没实例化
    if not st.session_state.get("thread_id"):
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"

    st.session_state.topic = sanitize_text(topic)
    st.session_state.progress = []
    st.session_state.report = ""
    st.session_state.answer = ""
    st.session_state.payload = {"topic": st.session_state.topic}
    st.session_state.phase = "running"


def _cb_resume() -> None:
    thread_id = (st.session_state.get("thread_id") or "").strip()
    if not thread_id:
        st.session_state.flash = "先在左侧填 thread_id"
        return

    saved = _get_graph().get_state({"configurable": {"thread_id": thread_id}})
    if not saved.values:
        st.session_state.flash = f"thread_id={thread_id!r} 上没有找到任何检查点"
        return

    st.session_state.topic = saved.values.get("topic", "")
    st.session_state.progress = []
    st.session_state.report = ""
    st.session_state.answer = ""
    st.session_state.payload = None       # None = 从检查点继续
    st.session_state.phase = "running"


def _cb_reset() -> None:
    for key, value in _DEFAULTS.items():
        st.session_state[key] = value


def _cb_submit_review() -> None:
    edited = (st.session_state.get("review_edited") or "").strip()
    reason = (st.session_state.get("review_reason") or "").strip()

    if reason:
        resume_value = f"no {reason}"
    else:
        resume_value = sanitize_text(edited)

    st.session_state.pending = None
    st.session_state.payload = Command(resume=resume_value)
    st.session_state.phase = "running"


def _cb_ask() -> None:
    question = (st.session_state.get("ask_input") or "").strip()
    if not question:
        st.session_state.flash = "先写一个问题"
        return

    st.session_state.answer = ""
    st.session_state.payload = {"followup_question": sanitize_text(question)}
    st.session_state.phase = "running"


# ============================================================================
# 推进图
# ============================================================================
def _drive() -> None:
    """把图往前推，直到遇到中断或跑完。

    做三件事：
        1. 节点进度写进 session_state.progress
        2. 报告正文的流式增量追加到 session_state.report
        3. 遇到中断就记下来，交给下一阶段渲染审核表单

    payload 传 None 是合法的 —— 表示「从检查点继续」。
    """
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": research_conf["recursion_limit"],
    }

    payload = st.session_state.payload
    st.session_state.payload = None

    status = st.status("正在运行…", expanded=True)
    report_box = st.empty()

    pending = None
    try:
        for chunk in graph.stream(
            payload, config, stream_mode=["updates", "custom"], subgraphs=True
        ):
            namespace, mode, data = _unpack(chunk)

            if mode == "custom":
                _handle_custom(data, status, report_box)
            elif mode == "updates" and not namespace:
                for node, update in data.items():
                    if node == "__interrupt__":
                        items = update if isinstance(update, (list, tuple)) else (update,)
                        pending = items[0] if items else None
                    elif not node.startswith("__"):
                        status.write(f"✓ {node} 执行完毕")
    except Exception as exc:
        status.update(label="运行失败", state="error")
        _show_root_error(exc)
        st.session_state.pending = None
        st.session_state.phase = "done"
        return

    st.session_state.pending = pending

    if pending is not None:
        status.update(label="等待人工审核", state="complete")
        st.session_state.phase = "review"
    else:
        status.update(label="完成", state="complete")
        _load_final_state(config)
        st.session_state.phase = "done"

    st.rerun()


def _unpack(chunk):
    """兼容 subgraphs 开关：二元组 or 三元组。"""
    if len(chunk) == 3:
        namespace, mode, data = chunk
        return namespace, mode, data
    mode, data = chunk
    return (), mode, data


def _handle_custom(data, status, report_box) -> None:
    """处理 get_stream_writer 推出来的两类消息。"""
    if isinstance(data, dict) and data.get("kind") == "report_delta":
        st.session_state.report += data.get("text", "")
        report_box.markdown(st.session_state.report + "▌")
        return

    st.session_state.progress.append(str(data))
    status.write(f"· {data}")


def _load_final_state(config: dict) -> None:
    """跑完之后从检查点读权威状态。

    为什么必须这么做：恢复运行时，planner 等节点不会出现在本次的 updates 流里
    （它们在上一次进程就跑完了）。只有检查点里永远装着完整状态。
    """
    values = _get_graph().get_state(config).values
    st.session_state.report = values.get("report") or st.session_state.report
    messages = values.get("messages") or []
    if messages and values.get("followup_question"):
        st.session_state.answer = getattr(messages[-1], "content", "")


def _show_root_error(exc: Exception) -> None:
    """把最内层的真实异常挖出来 —— LangGraph 的异常常被包好几层。"""
    root: BaseException = exc
    while root.__cause__ is not None:
        root = root.__cause__
    st.error(f"**{type(root).__name__}**\n\n```\n{root}\n```")


# ============================================================================
# 界面
# ============================================================================
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 会话")
        st.text_input(
            "thread_id（会话 ID）",
            key="thread_id",          # ← 这个 key 在本文件里只出现这一次
            help=(
                "同一个 ID = 同一条执行历史。"
                "刷新页面、重开浏览器后填同一个 ID 就能找回上次的研究。"
                "留空则自动生成。"
            ),
        )
        if not st.session_state.thread_id:
            st.caption("留空则在开始研究时自动生成")

        st.divider()
        st.markdown("### 当前配置")
        st.caption(f"检查点后端：**{research_conf.get('checkpointer')}**")
        st.caption(f"研究轮次上限：**{research_conf.get('max_iterations')}**")
        st.caption(f"并行度：**{research_conf.get('max_concurrency')}**")
        st.caption("改这些参数请编辑 `config/research.yml`")

        st.divider()
        st.markdown("### 状态")
        st.caption(f"阶段：`{st.session_state.phase}`")
        st.button("重置界面状态", use_container_width=True, on_click=_cb_reset)


def _render_idle() -> None:
    st.title("🔎 DeepResearch Agent")
    st.caption(
        "输入一个问题，自动拆解 → 并行检索 → 自我检查 → 输出带引用、可溯源的研究报告"
    )

    st.text_input(
        "研究主题",
        key="topic_input",
        placeholder="例如：RAG 与长上下文模型在知识问答场景下该如何选择",
    )

    col1, col2 = st.columns([1, 2])
    col1.button("开始研究", type="primary", use_container_width=True, on_click=_cb_start)
    col2.button(
        "恢复上次会话（需要左侧填 thread_id）",
        use_container_width=True,
        on_click=_cb_resume,
    )

    if st.session_state.flash:
        st.warning(st.session_state.flash)
        st.session_state.flash = ""


def _render_review() -> None:
    """审核面板 —— interrupt 暂停时出现。

    这是整个 UI 里最重要的部分：它把「图暂停了」变成一个用户能操作的界面。
    """
    st.title("⏸ 请审核研究计划")

    payload = getattr(st.session_state.pending, "value", None)
    if not isinstance(payload, dict):
        st.error("中断数据格式异常。")
        st.session_state.phase = "done"
        return

    st.markdown(f"**研究主题：** {payload.get('topic', '')}")
    st.info(payload.get("rationale", ""))

    st.markdown("**规划出的子问题：**")
    for index, question in enumerate(payload.get("sub_questions") or [], 1):
        st.markdown(f"{index}. {question}")

    st.divider()

    # 表单里的每个元素同样需要唯一的 key
    with st.form("review_form"):
        st.text_area(
            "如果你想调整，用逗号分隔写出新的子问题；不改就留空",
            key="review_edited",
            height=100,
            placeholder="欧洲市场政策, 关税影响, 本地化生产",
        )
        st.text_input(
            "如果要打回重做，在这里写理由（写了理由就会被打回）",
            key="review_reason",
            placeholder="太偏技术了，我想了解市场和政策",
        )
        st.form_submit_button("提交", type="primary", on_click=_cb_submit_review)


def _render_done() -> None:
    st.title("📄 研究报告")

    if st.session_state.flash:
        st.warning(st.session_state.flash)
        st.session_state.flash = ""

    if st.session_state.answer:
        st.markdown("### 追问回答")
        st.markdown(st.session_state.answer)
        st.divider()

    if not st.session_state.report:
        st.warning("这次运行没有产出报告。")
    else:
        st.markdown(st.session_state.report)
        st.download_button(
            "下载 Markdown",
            data=st.session_state.report,
            file_name=f"{st.session_state.topic or 'report'}.md",
            mime="text/markdown",
        )

    st.divider()
    st.markdown("### 追问")
    st.caption("基于已有的证据和报告回答，**不会触发任何检索**。")

    st.text_input("你的问题", key="ask_input", placeholder="用一句话给出选型建议？")
    st.button("提问", type="primary", on_click=_cb_ask)

    with st.expander("查看进度日志"):
        for line in st.session_state.progress:
            st.text(line)


# ============================================================================
# 主入口
# ============================================================================
def main() -> None:
    st.set_page_config(page_title="DeepResearch Agent", page_icon="🔎", layout="wide")
    setup_tracing()
    _init_state()
    _render_sidebar()

    phase = st.session_state.phase
    if phase == "idle":
        _render_idle()
    elif phase == "running":
        _drive()
    elif phase == "review":
        _render_review()
    else:
        _render_done()
