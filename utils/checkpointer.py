"""持久化检查点的工厂。

检查点（checkpoint）是 LangGraph 的「存档」机制：图的每一步执行完，
它会把当时的状态快照存下来。有了它才能做三件事：

    ① 断点续跑 —— 进程崩了 / 被 Ctrl+C，重启用同一个 thread_id 就能接着跑
    ② 人工审核 —— interrupt 暂停后，进程可以退出，之后用 Command(resume=...) 恢复
    ③ 多轮追问 —— 同一份研究结果上继续提问，不必重新检索

三者都依赖同一个概念：**thread_id**。它相当于「存档文件的名字」。
一个 thread_id 对应一条独立的执行历史。

===============================================================================
生产级 vs 开发态的选择
===============================================================================

    memory   进程内，零依赖。进程退出就没了。开发时用（默认）
    sqlite   落盘到 data/checkpoints/research.sqlite。演示 / 作品集用

为什么不做成每次调用都新建？
    Streamlit 每次交互都会重跑整个脚本。如果每次新建 saver，
    上一次会话的状态就找不回来了 —— 用户会看到「刚跑的东西突然没了」。

为什么不用 PostgresSaver？
    环境里装了 langgraph-checkpoint-postgres，但一个单机作品集项目
    引入数据库服务是自找麻烦（要装、要启、要配连接串）。
    面试时说明「生产可换 PostgresSaver，接口完全一致」即可 ——
    这正是 checkpointer 抽象的价值。
"""

import sqlite3
from functools import lru_cache
from pathlib import Path

from utils.config_handler import research_conf
from utils.path_tool import ensure_dir


@lru_cache(maxsize=1)
def get_checkpointer():
    """按 config/research.yml 里的 checkpointer 配置返回持久化后端。

    用 lru_cache 做成**单例**。原因见模块顶部关于 Streamlit 的说明。
    """
    kind = str(research_conf.get("checkpointer", "memory")).strip().lower()

    if kind == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    if kind == "sqlite":
        return _build_sqlite_saver()

    raise ValueError(
        f"config/research.yml 里的 checkpointer 只能是 memory 或 sqlite，"
        f"当前是 {kind!r}"
    )


def _build_sqlite_saver():
    """构建 SQLite 后端。

    有两个 Windows 上必踩的坑：

    ① **必须用 SqliteSaver(conn) 而不是 SqliteSaver.from_conn_string(...)**
       from_conn_string 返回的是一个上下文管理器，`with` 块一退出连接就关了。
       CLI 里可能还看不出问题（跑完就退出），但 Streamlit 每次交互重跑脚本，
       连接一关状态就丢 —— 表现为「刷新页面后历史报告不见了」，很难查。

    ② **check_same_thread=False**
       LangGraph 的并行分支跑在线程池里（M4 的 max_concurrency），
       而 sqlite3 默认禁止跨线程使用连接。不开这个开关，
       一旦有并行节点写检查点就会抛 ProgrammingError。
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(ensure_dir("data/checkpoints")) / "research.sqlite"

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    # WAL 模式：允许「一个写 + 多个读」并发，显著减少 database is locked。
    # 场景：CLI 正在跑，同时 Streamlit 页面在读历史。
    conn.execute("PRAGMA journal_mode=WAL")

    return SqliteSaver(conn)


def reset_checkpointer_cache() -> None:
    """清掉单例缓存。改完配置想立刻生效时用（正常流程用不到）。"""
    get_checkpointer.cache_clear()


if __name__ == "__main__":
    from utils.console import console, rule

    configured = research_conf.get("checkpointer", "memory")

    rule("检查点后端")
    console.print(f"  配置值 : [bold]{configured}[/bold]（改 config/research.yml 切换）")
    saver = get_checkpointer()
    console.print(f"  实例   : {type(saver).__name__}")

    if configured == "sqlite":
        path = Path("data/checkpoints/research.sqlite").resolve()
        console.print(f"  落盘位置: {path}")
        console.print(f"  文件存在: {path.exists()}")

    console.print(
        "\n  [dim]同一个 thread_id = 同一条执行历史。"
        "换 thread_id 就是开一个新的存档。[/dim]"
    )
