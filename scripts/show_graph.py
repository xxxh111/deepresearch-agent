"""打印当前图的结构（Mermaid 源码）。

    conda activate langgraph
    python scripts/show_graph.py

把输出粘到 https://mermaid.live 就能看图。
M8 会把这个输出自动同步进 README，保证图和代码不会脱节。

图是「随代码长出来」的 —— M1 只有 planner 一个节点，
到 M7 会长成带并行分支和循环回边的完整形态。每完成一个里程碑都值得回来看一眼。
"""

import sys
from pathlib import Path

# 直接 python scripts/show_graph.py 时，sys.path 里只有 scripts/ 没有项目根
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.console  # noqa: E402,F401  —— 副作用：修复中文编码
from graph import build_graph  # noqa: E402
from utils.console import console  # noqa: E402


def main() -> None:
    graph = build_graph()
    drawable = graph.get_graph()

    console.print("[bold]节点[/bold]")
    for node in drawable.nodes:
        console.print(f"  · {node}")

    console.print("\n[bold]边[/bold]")
    for edge in drawable.edges:
        arrow = "┄>" if edge.conditional else "->"
        label = f"  [dim]({edge.data})[/dim]" if edge.data else ""
        console.print(f"  {edge.source} {arrow} {edge.target}{label}")

    console.print("\n[bold]Mermaid 源码[/bold]（粘到 https://mermaid.live 看图）")
    console.print()
    print(drawable.draw_mermaid())


if __name__ == "__main__":
    main()
