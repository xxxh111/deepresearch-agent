"""LangSmith 可观测性配置。

LangSmith 完全靠环境变量驱动，不需要改代码。本模块做两件事：
    1. 补默认值（比如 LANGSMITH_PROJECT），让用户少配一点
    2. 提供一个可打印的状态报告，交给 scripts/check_env.py

国内网络注意：api.smith.langchain.com 可能需要代理。直连不通时把
LANGSMITH_TRACING 设为 false 即可 —— 图本身照常运行，只是少了 trace。
"""

import os

import utils.console  # noqa: F401  —— 副作用：触发编码修复
from utils.config_handler import llm_conf  # noqa: F401  —— 副作用：加载 .env

_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
_DEFAULT_PROJECT = "deepresearch-agent"


def setup_tracing(project: str | None = None) -> bool:
    """补齐 LangSmith 环境变量默认值。返回是否已启用 tracing。"""
    os.environ.setdefault("LANGSMITH_ENDPOINT", _DEFAULT_ENDPOINT)
    os.environ.setdefault("LANGSMITH_PROJECT", project or _DEFAULT_PROJECT)

    # 没配 key 就别开 tracing，否则 langsmith 会一直重试并刷警告。
    # 注意这里用 is_configured 而不是 bool() —— 从 .env.example 复制来的
    # 占位符（lsv2_xxxx...）是非空字符串，bool() 会判成「已配置」，
    # 于是 tracing 被打开然后一路 401。占位符必须被识别出来。
    enabled = _is_truthy(os.getenv("LANGSMITH_TRACING")) and is_configured(
        os.getenv("LANGSMITH_API_KEY")
    )
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
    return enabled


def tracing_status() -> dict:
    """给 check_env 用的状态报告（不泄露明文 key）。"""
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    return {
        "enabled": _is_truthy(os.getenv("LANGSMITH_TRACING")),
        "project": os.getenv("LANGSMITH_PROJECT", _DEFAULT_PROJECT),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", _DEFAULT_ENDPOINT),
        "api_key": mask_secret(api_key),
    }


def mask_secret(secret: str) -> str:
    """只露出前 8 位 —— 够确认「填没填、填对没」就行，绝不打印完整密钥。"""
    if not secret:
        return "（未设置）"
    if not is_configured(secret):
        return "（仍是 .env.example 的占位符）"
    return f"{secret[:8]}...（长度 {len(secret)}）"


# 占位符特征：.env.example 里全是 sk-xxxx / tvly-xxxx / lsv2_xxxx 这种
_PLACEHOLDER_MARKERS = ("xxxx", "your_", "your-", "<", "changeme")


def is_configured(value: str | None) -> bool:
    """判断一个密钥是否真的配置了。

    单纯判空是不够的：用户从 .env.example 复制出 .env 后，
    字段是 `DEEPSEEK_API_KEY=sk-xxxxxxxx...` —— 非空，但完全没用。
    这种情况必须识别成「未配置」，否则错误会推迟到第一次调用模型时才暴露。
    """
    if not value or not value.strip():
        return False
    lowered = value.strip().lower()
    return not any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes", "on"}
