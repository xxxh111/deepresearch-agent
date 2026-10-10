"""检索工具：把 Tavily 包成给 LLM 用的工具。

【为什么要包一层，而不是直接把 TavilySearch 塞给模型】
    1. 降级 —— 配额耗尽、网络不通时返回一句「我失败了」比抛异常好得多。
       抛异常会毁掉整轮运行，而前面已经花掉的检索额度全白费。
    2. 输出归一化 —— Tavily 返回嵌套 dict，模型读起来费劲还容易漏字段。
       压成一段结构化文本，模型的工具调用准确率明显更高。
    3. 参数控制 —— chunks_per_source 这类关键参数只有包一层才好统一管理。

【@tool 的 docstring 就是工具描述】
    它会原样发给模型，供模型判断「什么时候该用这个工具」。
    这和 M2 里 Field(description=...) 是一模一样的机制 ——
    写得含糊，模型就乱调工具。所以下面的 docstring 都是当提示词写的。

【实测得到的两条事实】
    · TavilySearch  缺 API key 时构造器**静默通过**，调用时才炸
    · TavilyExtract 缺 API key 时构造器**直接抛异常**
    两者不一致，所以不能靠「能不能构造成功」判断 key 配好了没 —— 必须显式检查。
"""

import os
import time
from functools import lru_cache
from typing import Any, Callable

from langchain_core.tools import tool
from langchain_tavily import TavilyExtract, TavilySearch

from utils.config_handler import search_conf
from utils.console import console
from utils.path_tool import get_abs_path
from utils.tracing import is_configured


# ============================================================================
# 前置检查与降级
# ============================================================================
def _require_api_key() -> None:
    """fail-fast：key 没配就立刻说清楚，不要等跑到一半才炸。"""
    if is_configured(os.getenv("TAVILY_API_KEY")):
        return
    raise RuntimeError(
        "TAVILY_API_KEY 未配置或仍是占位符。\n"
        "  1. 去 https://app.tavily.com/ 注册（免费 1000 credits/月）\n"
        f"  2. 把 key 填进 {get_abs_path('.env')} ：TAVILY_API_KEY=tvly-...\n"
        "  3. 跑 python scripts/check_env.py 确认"
    )


def _with_retry(call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any | None:
    """指数退避重试。全部失败返回 None，交给调用方决定怎么降级。

    只在这里兜住「工具调用失败」，不吞掉其他逻辑错误 ——
    所以调用方必须先判断返回值是不是 None。
    """
    attempts = int(search_conf.get("retry_attempts", 2)) + 1
    delay = 1.0
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return call(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 —— 工具层要兜住所有失败并降级
            last_error = exc
            if attempt < attempts:
                console.print(
                    f"[yellow]  工具调用失败（{type(exc).__name__}），"
                    f"{delay:.0f}s 后重试（{attempt}/{attempts - 1}）[/yellow]"
                )
                time.sleep(delay)
                delay *= 2

    console.print(f"[red]  工具调用重试 {attempts} 次后仍失败：{last_error}[/red]")
    return None


def _extract_failure_reason(result: Any) -> str | None:
    """把 Tavily 各种「失败形态」归一成一句人能读懂的原因；成功则返回 None。

    【为什么需要这个函数 —— 实测出来的】
    TavilyExtract 的返回类型**不可靠**，实测到三种：

        dict + "results"   → 成功，正常解析
        dict + "error"     → 失败。例如 {'error': ValueError('All URLs failed validation.')}
        str                → 失败。例如 "No extracted results found for '[...]'. Suggestions: ..."

    第三种最阴险：它在**类型层面**就是个字符串，代码里写 result["results"] 会直接
    AttributeError。而且它在「页面存在但抽取不出内容」时也会出现 —— 不是异常，
    是正常返回一个字符串。

    写第三方 API 封装时，永远不要假设返回类型只有一种。
    """
    if result is None:
        return "重试后仍然失败（网络问题或配额耗尽）"

    if isinstance(result, str):
        # Tavily 把错误说明直接当字符串返回了
        return " ".join(result.split())[:200]

    if isinstance(result, dict) and result.get("error"):
        return str(result["error"])[:200]

    return None


# ============================================================================
# Tavily 客户端（惰性 + 缓存）
# ============================================================================
# 做成 lru_cache 工厂而不是模块级常量，原因和 model/factory.py 一样：
# 模块级构造意味着 import 这个文件就会去创建客户端，
# 缺 key 时连 check_env 和单元测试都跑不起来。
# 惰性构造保证 import 永远安全。
@lru_cache(maxsize=1)
def _searcher() -> TavilySearch:
    _require_api_key()
    return TavilySearch(
        max_results=int(search_conf["max_results"]),
        search_depth=search_conf["search_depth"],
        topic=search_conf["topic"],
        # raw_content 保持 False：我们有更省钱的 fetch_page 去读正文，
        # 搜索阶段只要摘要就够了。
        include_raw_content=False,
    )


def _build_extractor(query: str) -> TavilyExtract:
    """每次调用都新建 —— 因为 query 是按本次调用变化的。

    query 必须传：Tavily 会按它挑选最相关的片段。
    不传的话返回整页 HTML 转换结果（含导航栏、侧栏、评论区）。
    实测对比（同一个 CSDN 页面）：
        不传 query                → 54,085 字符，开头是"CSDN首页 博客 下载 社区"
        query + chunks_per_source →  2,106 字符，开头就是正文
    """
    _require_api_key()
    extract_conf = search_conf.get("extract", {})
    return TavilyExtract(
        extract_depth=extract_conf.get("extract_depth", "basic"),
        format=extract_conf.get("format", "markdown"),
        query=query,
        chunks_per_source=int(extract_conf.get("chunks_per_source", 3)),
    )


# ============================================================================
# 工具 1：搜索
# ============================================================================
@tool
def web_search(query: str) -> str:
    """搜索互联网，返回若干条最相关的网页（标题、链接、摘要）。

    这是获取信息的**第一步**：先用它找到候选页面。
    返回结果里的 URL 可以传给 fetch_page 去读取正文。
    如果摘要已经足够回答问题，就不必再调用 fetch_page —— 每次读取都要花额度。

    Args:
        query: 检索词。用与用户提问相同的语言，写成完整问句效果更好。
    """
    _require_api_key()

    result = _with_retry(_searcher().invoke, {"query": query})
    failure = _extract_failure_reason(result)
    if failure:
        return (
            f"搜索失败：{failure}\n"
            "请基于你已有的知识继续，并在结论中说明这一部分未能检索到最新资料。"
        )

    items = result.get("results") or []
    if not items:
        return f"没有找到与「{query}」相关的网页。换一种说法再搜一次可能会有结果。"

    lines = [f"检索词：{query}", f"找到 {len(items)} 条结果：", ""]
    for index, item in enumerate(items, 1):
        title = (item.get("title") or "(无标题)").strip()
        url = (item.get("url") or "").strip()
        snippet = " ".join((item.get("content") or "").split())
        lines.append(f"[{index}] {title}")
        lines.append(f"    URL: {url}")
        lines.append(f"    摘要: {snippet[:300]}")
        lines.append("")

    lines.append("需要读某个页面的正文时，调用 fetch_page(url=..., query=...)")
    return "\n".join(lines)


# ============================================================================
# 工具 2：读取网页正文
# ============================================================================
@tool
def fetch_page(url: str, query: str) -> str:
    """读取指定网页中与你关心的内容最相关的正文片段。

    必须在 web_search 之后使用 —— 先搜索拿到可用的 URL，再用本工具读取。
    本工具只返回与 query 相关的片段，不返回整页，所以 query 写得越具体越有用。

    Args:
        url: 要读取的网页地址，必须来自 web_search 的返回结果。
        query: 你想从这个页面里找什么，写成具体的问题或关键词。
            例如「检查点的存储结构和序列化格式」比「检查点」好得多。
    """
    _require_api_key()

    result = _with_retry(_build_extractor(query).invoke, {"urls": [url]})
    failure = _extract_failure_reason(result)
    if failure:
        return (
            f"无法读取 {url}：{failure}\n"
            "可以换一个 URL 试试，或基于搜索摘要继续。"
        )

    items = result.get("results") or []
    if not items:
        failed = result.get("failed_results") or []
        reason = failed[0].get("error") if failed else "页面无法解析"
        return f"无法读取 {url}：{reason}。换其他 URL 试试。"

    item = items[0]
    title = (item.get("title") or "(无标题)").strip()
    content = (item.get("raw_content") or "").strip()

    if not content:
        return f"页面 {url} 没有可提取的正文内容。"

    # 兜底护栏：chunks_per_source 已经控制住长度了，
    # 这一步只是防止某个页面异常膨胀把上下文撑爆。
    max_chars = int(search_conf.get("extract", {}).get("max_chars_per_page", 8000))
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    header = f"来源：{title}\nURL：{url}\n"
    footer = "\n\n（内容过长已截断）" if truncated else ""
    return f"{header}\n{content}{footer}"


# 供 bind_tools / ToolNode 使用
ALL_TOOLS = [web_search, fetch_page]


# ============================================================================
# 自测：不经过图，单独验证这两个工具
# ============================================================================
if __name__ == "__main__":
    from utils.console import rule

    rule("工具自测（不经过 LangGraph）")

    console.print("[bold]① web_search[/bold]")
    output = web_search.invoke({"query": "LangGraph 检查点 Checkpointer 机制"})
    console.print(output[:900])
    console.print("[dim]…（省略）[/dim]")

    console.print("\n[bold]② fetch_page[/bold]")
    # 从上面结果里挑一个 URL —— 这里写死一个，你也可以换成上面输出里的任意一个
    demo_url = "https://blog.csdn.net/u013172930/article/details/148036704"
    page = fetch_page.invoke(
        {"url": demo_url, "query": "检查点的存储结构和序列化格式"}
    )
    console.print(page[:900])
    console.print("[dim]…（省略）[/dim]")

    console.print("\n[bold]③ 工具描述（这就是模型看到的内容）[/bold]")
    for t in ALL_TOOLS:
        console.print(f"  [cyan]{t.name}[/cyan]: {t.description.splitlines()[0]}")
