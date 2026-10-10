"""配置加载。

沿用你 Agent项目 的「YAML → 模块级全局变量」模式，两点改进：
    1. 所有 open() 显式带 encoding="utf-8"（不加的话 Windows 下读中文 YAML 会炸）
    2. 在这里 load_dotenv()，保证 .env 在任何 langchain/langsmith 代码运行前就位

用法：
    from utils.config_handler import research_conf
    print(research_conf["max_iterations"])
"""

import yaml
from dotenv import load_dotenv

from utils.path_tool import get_abs_path

# 必须在任何 langchain / langsmith 客户端被创建之前执行。
# override=True：.env 里的值优先于系统环境变量（和你 langgraph 学习笔记里的写法一致）。
load_dotenv(get_abs_path(".env"), override=True)


def _load_yaml(name: str) -> dict:
    with open(get_abs_path(f"config/{name}.yml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


llm_conf = _load_yaml("llm")
research_conf = _load_yaml("research")
search_conf = _load_yaml("search")
prompts_conf = _load_yaml("prompts")


if __name__ == "__main__":
    from utils.console import console

    console.print("llm_conf      :", llm_conf)
    console.print("research_conf :", research_conf)
    console.print("search_conf   :", search_conf)
    console.print("prompts_conf  :", prompts_conf)
