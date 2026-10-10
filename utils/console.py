"""控制台输出 + Windows 中文编码修复。

【重要】导入本模块会**全进程**修复 stdout/stderr 编码，所以任何要打印中文的
模块都应该（直接或间接）先导入它。

背景（实测复现）：
    在 D:\\anaconda\\envs\\langgraph 下直接 print('中文测试') 会输出
    `���Ĳ��ԣ��������`。原因是 Python 在 Windows 上按控制台代码页（GBK/CP936）
    编码输出，而终端按 UTF-8 解码。

修复方式：
    sys.stdout.reconfigure(encoding='utf-8')
    —— 只影响本进程，不改系统设置，也不需要设 PYTHONUTF8=1。
"""

import sys

from rich.console import Console


def _force_utf8() -> None:
    """把标准输入/输出/错误流全部强制切成 UTF-8。

    【为什么 stdin 也要改 —— 这是后来补的】
    最初只改 stdout/stderr，以为够了。结果人工审核那条流程直接崩：

        UnicodeEncodeError: 'utf-8' codec can't encode character '\\udcaa'

    原因是 Windows 上 Python 默认按控制台代码页（GBK）读 stdin，
    而管道/终端送进来的其实是 UTF-8 字节 —— 解码错位会产生**代理字符**
    （surrogate，形如 \\udcaa）。它钻进图状态之后，写 sqlite 检查点时要编码成
    UTF-8，代理字符不允许出现在 UTF-8 里，于是整轮运行崩掉。

    教训：**编码是三件事，不是两件。** 只修输出不修输入，
    错误的数据会从你没想到的入口进来，然后在很远的地方爆炸。
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


_force_utf8()


def sanitize_text(text: str) -> str:
    """清掉字符串里的代理字符，返回干净的 UTF-8 文本。

    这是**兜底**，不是主要防线 —— 主要防线是上面的 _force_utf8()。
    但用户输入来自很多渠道（终端、管道、以后的 Streamlit），
    任何一条路出问题都不该让整轮研究崩掉。

    原理：用 errors="ignore" 编码成 UTF-8，编码不了的字符会被丢掉，
    再解回来就是干净字符串。
    """
    if not text:
        return text
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

# 全局单例。rich 在构造时会读取 sys.stdout.encoding —— 所以必须在 _force_utf8() 之后建。
console = Console()


def rule(title: str = "") -> None:
    """打一条分隔线，用于 CLI 里区分阶段。"""
    console.rule(title, style="dim")


def emit(message: str) -> None:
    """节点内往前端推进度。

    在 LangGraph 节点里调用 get_stream_writer() 拿到的写入器可以把任意对象
    推进 stream_mode="custom" 流。不在图执行上下文里调用时它返回空操作，所以
    这里直接 try 兜底，避免单测/脚本里裸调用报错。

    注意导入路径：langgraph.config，不是 langgraph.types（网上博客常写错）。
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(message)
    except Exception:
        pass
