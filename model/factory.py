"""模型工厂：provider 注册表 + 角色化参数。

对比你 Agent项目 的 model/factory.py，两个关键改进：

1. **惰性构造**
   原版在模块顶层写 `chat_model = ChatModelFactory().generator()`，
   意味着只要 import 这个模块就会真的去创建模型客户端 —— 而 langchain-deepseek
   在缺 DEEPSEEK_API_KEY 时会直接抛 ValidationError。结果就是「我只想 import 一下
   配置」也会炸，单元测试和 check_env 这类脚本没法用。
   这里改成**调用时才创建**，import 永远安全。

2. **provider 可插拔 + 角色化温度**
   provider 从 config/llm.yml 读（可被环境变量 LLM_PROVIDER 覆盖）。
   不同角色给不同 temperature：planner 要 0.0（要稳定的 JSON），
   writer 要 0.4（要可读性）。这是低成本但很能说明工程意识的细节。

用法：
    from model.factory import get_chat_model
    planner_llm = get_chat_model("planner")
    writer_llm = get_chat_model("writer")
"""

import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from utils.config_handler import llm_conf


def get_provider() -> str:
    """当前 provider。环境变量 LLM_PROVIDER 优先于 config/llm.yml。"""
    return (os.getenv("LLM_PROVIDER") or llm_conf["provider"]).strip().lower()


def _build_deepseek(role_conf: dict) -> BaseChatModel:
    from langchain_deepseek import ChatDeepSeek

    cfg = llm_conf["providers"]["deepseek"]
    kwargs: dict = {
        "model": cfg["model"],
        "temperature": role_conf.get("temperature", 0.0),
        "max_retries": cfg.get("max_retries", 2),
    }
    # 关键：关闭思考模式。
    # 开着的话结构化输出会落进 reasoning_content 而 content 为空，
    # with_structured_output 直接解析失败。你的 06 notebook 已经这么干了，沿用。
    if cfg.get("thinking") == "disabled":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatDeepSeek(**kwargs)


def _build_qwen(role_conf: dict) -> BaseChatModel:
    # 走 OpenAI 兼容端点，而不是 langchain_community.ChatTongyi ——
    # 后者已在弃用轨道上（langchain-community 处于维护模式）。
    from langchain_openai import ChatOpenAI

    cfg = llm_conf["providers"]["qwen"]
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=os.getenv(cfg.get("api_key_env", "DASHSCOPE_API_KEY")),
        temperature=role_conf.get("temperature", 0.0),
        max_retries=cfg.get("max_retries", 2),
    )


def _build_openai(role_conf: dict) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    cfg = llm_conf["providers"]["openai"]
    return ChatOpenAI(
        model=cfg["model"],
        temperature=role_conf.get("temperature", 0.0),
        max_retries=cfg.get("max_retries", 2),
    )


_REGISTRY = {
    "deepseek": _build_deepseek,
    "qwen": _build_qwen,
    "openai": _build_openai,
}


@lru_cache(maxsize=None)
def get_chat_model(role: str = "planner") -> BaseChatModel:
    """按角色取模型。同一角色重复调用返回同一个实例（模型客户端可复用）。"""
    provider = get_provider()
    if provider not in _REGISTRY:
        raise ValueError(
            f"未知的 LLM_PROVIDER: {provider!r}。可选：{sorted(_REGISTRY)}"
        )
    role_conf = llm_conf.get("roles", {}).get(role, {})
    return _REGISTRY[provider](role_conf)


if __name__ == "__main__":
    from utils.console import console

    console.print(f"当前 provider: [bold]{get_provider()}[/bold]")
    console.print(f"roles 配置    : {sorted(llm_conf.get('roles', {}))}")
    console.print(
        "\n[dim]注意：import 本模块不会创建任何模型客户端。"
        "只有调用 get_chat_model() 才会 —— 所以 check_env 和测试可以先跑。[/dim]"
    )
