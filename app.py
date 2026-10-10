"""Streamlit 入口（薄壳）。

    conda activate langgraph
    streamlit run app.py

真正的界面代码在 ui/streamlit_app.py。
这层壳存在的原因：Streamlit 会把入口脚本的名字当作应用名，
而且程序化启动（比如 `python app.py` 或部署时）需要一个明确的落点。
"""

import sys
from pathlib import Path

# Streamlit 运行时会把入口脚本所在目录加进 sys.path，
# 但显式加一次更稳（比如从别的目录用绝对路径启动时）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.streamlit_app import main  # noqa: E402

main()
