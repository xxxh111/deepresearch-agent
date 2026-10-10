"""图装配包。

对外只暴露 build_graph —— 调用方不需要知道内部有多少节点。
"""

from graph.build import build_graph

__all__ = ["build_graph"]
