"""提示词加载：名字 → prompts/*.txt 内容。

提示词全部外置，代码里不出现长字符串。带 lru_cache，同一提示词只读一次盘。

用法：
    from utils.prompt_loader import load_prompt
    system = load_prompt("planner")
"""

from functools import lru_cache

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """按 config/prompts.yml 里的键名加载提示词全文。"""
    if name not in prompts_conf:
        raise KeyError(
            f"prompts.yml 里没有 {name!r} 这个键。现有键：{sorted(prompts_conf)}"
        )
    with open(get_abs_path(prompts_conf[name]), "r", encoding="utf-8") as f:
        return f.read()


def prompt_names() -> list[str]:
    """所有已注册的提示词名。"""
    return sorted(prompts_conf)


if __name__ == "__main__":
    from utils.console import console

    for n in prompt_names():
        try:
            text = load_prompt(n)
            console.print(f"[green]OK[/green]   {n:<20} {len(text):>5} 字符")
        except FileNotFoundError:
            console.print(f"[yellow]缺失[/yellow] {n:<20} （尚未创建）")
