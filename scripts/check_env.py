"""启动前自检。动手写代码之前先跑这个，一次性暴露所有环境问题。

    conda activate langgraph
    python scripts/check_env.py

检查五项：
    1. Python 版本
    2. 关键依赖版本
    3. LangGraph 1.x 的 API 是否真的存在（防止环境里装的是 0.x 老版本）
    4. 三个 API key 是否配好（只显示掩码，绝不打印明文）
    5. Windows 中文编码是否正常

【为什么必须有这个脚本】
TavilySearch 在缺少 TAVILY_API_KEY 时**构造器不报错** —— 要等真正发起搜索
才炸，那时流程已经跑到一半、额度也可能已经花了。所以在入口处主动 fail-fast。
"""

import os
import sys
from pathlib import Path

# 让脚本能 import 到项目根的 utils/ ——
# 直接 python scripts/check_env.py 时，sys.path 里只有 scripts/ 没有项目根
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import platform  # noqa: E402

import utils.console  # noqa: E402,F401  —— 副作用：修复中文编码
from utils.console import console  # noqa: E402
from utils.tracing import is_configured, setup_tracing, tracing_status  # noqa: E402

OK = "[green]✓[/green]"
BAD = "[red]✗[/red]"
WARN = "[yellow]![/yellow]"

# 期望的最低版本（元组比较，不用 packaging）
MIN_PYTHON = (3, 11)
REQUIRED_PACKAGES = {
    "langgraph": "1.1.2",
    "langchain": "1.2.12",
    "langchain-core": "1.2.18",
    "langchain-deepseek": "1.0.1",
    "langchain-tavily": "0.2.17",
    "langgraph-checkpoint": None,       # 版本不敏感，装了就行
    "langsmith": None,
    "pydantic": None,
}

problems: list[str] = []
warnings: list[str] = []


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold]{title}[/bold]", style="dim")


def check_python() -> None:
    section("1. Python 运行时")
    version = sys.version_info
    console.print(f"  版本    : {platform.python_version()}")
    console.print(f"  解释器  : {sys.executable}")
    console.print(f"  平台    : {platform.system()} {platform.release()}")

    if (version.major, version.minor) >= MIN_PYTHON:
        console.print(f"  {OK} 版本满足要求（>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}）")
    else:
        console.print(f"  {BAD} 版本过低，需要 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
        problems.append("Python 版本过低")

    # 顺带提醒环境是否选对了
    if "langgraph" not in sys.executable.lower():
        warnings.append(
            f"当前解释器不是 langgraph 环境：{sys.executable}\n"
            f"    建议先执行 conda activate langgraph"
        )


def check_packages() -> None:
    section("2. 关键依赖")
    from importlib.metadata import PackageNotFoundError, version

    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            console.print(f"  {BAD} {package:<28} 未安装")
            problems.append(f"缺少依赖 {package}")
            continue

        if expected is None:
            console.print(f"  {OK} {package:<28} {actual}")
        elif actual.startswith(expected):
            console.print(f"  {OK} {package:<28} {actual}")
        else:
            console.print(
                f"  {WARN} {package:<28} {actual}（预期 {expected}，"
                "小版本差异通常无碍）"
            )
            warnings.append(f"{package} 版本为 {actual}，预期 {expected}")


def check_langgraph_api() -> None:
    """确认环境里的 LangGraph 是 1.x，而不是网上教程里的 0.x。"""
    section("3. LangGraph 1.x API")

    checks = [
        ("langgraph.graph", ["StateGraph", "START", "END"]),
        ("langgraph.graph.message", ["add_messages"]),
        # 1.0 起 MemorySaver 改名为 InMemorySaver
        ("langgraph.checkpoint.memory", ["InMemorySaver"]),
        # 1.0 起 get_stream_writer 在 langgraph.config，不在 langgraph.types
        ("langgraph.config", ["get_stream_writer"]),
        ("langgraph.types", ["Send", "interrupt", "Command", "Overwrite"]),
        ("langgraph.prebuilt", ["ToolNode", "tools_condition"]),
    ]

    for module_name, names in checks:
        try:
            module = __import__(module_name, fromlist=names)
        except ImportError as exc:
            console.print(f"  {BAD} {module_name:<32} 导入失败：{exc}")
            problems.append(f"无法导入 {module_name}")
            continue

        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            console.print(f"  {BAD} {module_name:<32} 缺少 {missing}")
            problems.append(f"{module_name} 缺少 {missing}")
        else:
            console.print(f"  {OK} {module_name:<32} {', '.join(names)}")

    # 反面验证：确认网上博客常写错的路径确实是错的
    try:
        from langgraph.types import get_stream_writer  # noqa: F401

        console.print(
            f"  {WARN} langgraph.types 里竟然也有 get_stream_writer —— "
            "请以 langgraph.config 为准"
        )
    except ImportError:
        console.print(
            f"  {OK} 反面验证：langgraph.types 里[b]没有[/b] get_stream_writer"
        )
        console.print("     [dim]→ 所以导入路径必须是 langgraph.config[/dim]")


def check_keys() -> None:
    section("4. API Keys")

    setup_tracing()
    status = tracing_status()

    items = [
        ("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""), True, "默认模型 provider"),
        ("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""), True, "联网搜索"),
        ("LANGSMITH_API_KEY", os.getenv("LANGSMITH_API_KEY", ""), False, "可观测性"),
        ("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""), False, "仅 qwen provider 需要"),
    ]

    for name, value, required, purpose in items:
        configured = is_configured(value)
        shown = (value[:8] + "…") if configured else "（未设置）"
        if configured:
            console.print(f"  {OK} {name:<20} {shown:<14} [dim]{purpose}[/dim]")
        elif required:
            console.print(f"  {BAD} {name:<20} {shown:<14} [dim]{purpose}[/dim]")
            problems.append(f"{name} 未配置")
        else:
            console.print(f"  {WARN} {name:<20} {shown:<14} [dim]{purpose}[/dim]")

    # Tavily 特别提醒：它的构造器在缺 key 时不报错
    if not is_configured(os.getenv("TAVILY_API_KEY", "")):
        console.print(
            "\n  [yellow]重点提醒[/yellow]：TavilySearch 缺 key 时构造器不报错，"
            "要等真正搜索才炸。"
        )
        console.print("           所以这个检查很有必要，否则你会跑到一半才发现。")

    console.print(
        f"\n  LangSmith : "
        + (
            f"[green]已启用[/green] project={status['project']}"
            if status["enabled"]
            else "[dim]未启用（不影响图运行，只是没有 trace）[/dim]"
        )
    )


def check_encoding() -> None:
    section("5. Windows 中文编码")

    stdout_enc = (sys.stdout.encoding or "").lower()
    console.print(f"  sys.stdout.encoding : {stdout_enc}")

    if "utf" in stdout_enc:
        console.print(f"  {OK} 标准输出已是 UTF-8，中文不会乱码")
    else:
        console.print(
            f"  {BAD} 编码为 {stdout_enc}，中文会输出成乱码\n"
            "        修复：确保先 import utils.console（它调用 sys.stdout.reconfigure）"
        )
        problems.append("标准输出编码不是 UTF-8")

    # 真实写一次中文，肉眼确认。
    # 这里刻意用原生 print 而不是 console.print —— 日志 handler 走的就是 print 路径，
    # 要验证的是那条路径也正常。
    print("  原生 print 测试：你好世界 — 「引号」破折号—— emoji ✅")


def main() -> int:
    console.print("\n[bold]DeepResearch Agent —— 环境自检[/bold]")

    check_python()
    check_packages()
    check_langgraph_api()
    check_keys()
    check_encoding()

    section("结论")
    if problems:
        console.print(f"[bold red]发现 {len(problems)} 个必须解决的问题：[/bold red]")
        for p in problems:
            console.print(f"  {BAD} {p}")
    if warnings:
        console.print(f"\n[bold yellow]{len(warnings)} 条提醒：[/bold yellow]")
        for w in warnings:
            console.print(f"  {WARN} {w}")

    if not problems:
        console.print(f"\n{OK} [bold green]全部通过，可以开始跑图了。[/bold green]")
        console.print('  试试： python cli.py "为什么多智能体架构比单智能体更适合深度研究任务"')
        return 0

    console.print("\n[dim]提示：把 .env.example 复制成 .env 并填入真实密钥[/dim]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
