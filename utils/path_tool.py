"""路径工具。

全部路径都从这里出，代码里不写死 `D:\\...`、也不混用 `/` 和 `\\`。
（沿用你 Agent项目 的 get_abs_path 约定）
"""

from pathlib import Path


def get_project_root() -> Path:
    """项目根目录（本文件在 <root>/utils/path_tool.py，所以往上两级）。"""
    return Path(__file__).resolve().parent.parent


def get_abs_path(relative: str) -> str:
    """把相对项目根的路径转成绝对路径字符串。"""
    return str(get_project_root() / relative)


def ensure_dir(relative: str) -> str:
    """确保目录存在，返回绝对路径。"""
    path = get_project_root() / relative
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
