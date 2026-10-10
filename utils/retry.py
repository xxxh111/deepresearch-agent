"""LLM 节点的重试策略。

【为什么单独一个文件，不写在 graph/build.py 里】
planner（M2）、researcher（M3）、gap_checker / writer（M7）都要用同一套重试判断。
写在 build.py 里会越堆越乱，而那个文件的职责是「装配」，不是「定义策略」。
放这里任何模块都能 import，以后要调整规则也只改一处。

===============================================================================
这个文件存在的根本原因：LangGraph 的默认重试策略在我们的核心场景下是失效的
===============================================================================

langgraph.types.default_retry_on 的源码里有这么一段：

    if isinstance(exc, (ValueError, TypeError, ArithmeticError, ImportError,
                        LookupError, NameError, SyntaxError, RuntimeError,
                        ReferenceError, StopIteration, StopAsyncIteration, OSError)):
        return False

原意很合理：ValueError / TypeError 这类异常通常表示**你的代码写错了**，
重试一百次也是同样的错，纯浪费时间。

但它没考虑到 —— pydantic.ValidationError 恰好继承自 ValueError：

    pydantic_core.ValidationError  →  builtins.ValueError  →  Exception

而 ValidationError 的语义和普通 ValueError 完全不同：它表示
**模型输出不符合我们的 schema**。模型是有随机性的，下一轮很可能就改对了 ——
这正是最值得重试的情况，却落进了「不重试」名单。

实测结果（完整验证见 learn/02b_retry_policy.py）：

    ValueError       + RetryPolicy(max_attempts=3)  → 只调用 1 次，不重试
    ValidationError  + RetryPolicy(max_attempts=3)  → 只调用 1 次，不重试  ← 静默失效
    ConnectionError  + RetryPolicy(max_attempts=3)  → 调用 3 次，成功
    ValidationError  + 本文件的策略                  → 调用 3 次，成功

危害在于「静默」—— `add_node(..., retry_policy=RetryPolicy(max_attempts=3))`
看起来无懈可击，不报错、不警告，但那个重试保护根本不存在。

===============================================================================
另外两个默认策略的盲区（顺手一起修了）
===============================================================================

1. TimeoutError **不**重试
   因为 Python 3.3 起 TimeoutError 继承自 OSError（PEP 3151 的历史包袱），
   于是也进了不重试名单。可超时是最典型的临时故障，最该重试。
   （注意 openai.APITimeoutError 不受影响 —— 它继承 openai.APIError，会走兜底重试）

2. openai.AuthenticationError (401) **会**重试
   default_retry_on 最后一行是 `return True` ——「不认识的异常一律重试」。
   401 表示你的 API key 是错的，重试 3 次只是白等 + 刷屏。
"""

from langgraph.types import RetryPolicy, default_retry_on
from pydantic import ValidationError


def _invalid_request_errors() -> tuple[type[Exception], ...]:
    """「重试也没用」的请求类错误 —— 请求本身有问题，不是服务端的临时故障。

    延迟 import openai 的原因：utils.retry 被 import 时不必立刻拉进整个
    openai SDK。真正在用 ChatDeepSeek 时它本来就会加载，这里只是不抢先。
    """
    import openai

    return (
        openai.AuthenticationError,    # 401 API key 无效
        openai.PermissionDeniedError,  # 403 没有权限
        openai.NotFoundError,          # 404 模型名写错了
        openai.BadRequestError,        # 400 请求参数本身不合法
    )


def retry_on_llm_error(exc: Exception) -> bool:
    """判断一个异常值不值得重试。

    三层判断，顺序不能颠倒：
        1. 模型输出不合 schema  → 重试（本文件存在的理由）
        2. 请求本身有问题       → 不重试（重试一百次结果一样）
        3. 其余                 → 交给 LangGraph 的默认判断
                                  （网络故障、429、5xx 会重试；代码 bug 不会）
    """
    # 第 1 层：模型输出不符合 schema。
    # 必须放在最前面 —— 否则它会被下面的 default_retry_on 按 ValueError 处理掉。
    if isinstance(exc, ValidationError):
        return True

    # 第 2 层：鉴权 / 请求错误。重试只是浪费时间，而且掩盖了真正的问题。
    if isinstance(exc, _invalid_request_errors()):
        return False

    # 第 3 层：兜底。
    return default_retry_on(exc)


def llm_retry_policy(max_attempts: int = 3) -> RetryPolicy:
    """LLM 节点的标准重试策略。

    做成工厂函数而不是模块级常量：每次返回新实例，避免多个节点共享同一个
    可变对象（RetryPolicy 是 pydantic BaseModel，理论上可以被改）。

    用法：
        builder.add_node("planner", planner_node, retry_policy=llm_retry_policy())
    """
    return RetryPolicy(max_attempts=max_attempts, retry_on=retry_on_llm_error)


if __name__ == "__main__":
    import openai
    import httpx

    from utils.console import console, rule

    rule("retry_on_llm_error 的判断结果")

    class _Tiny:
        pass

    try:
        from pydantic import BaseModel, Field

        class _M(BaseModel):
            x: list[str] = Field(min_length=3)

        _M(x=[])
        validation_error: Exception = RuntimeError("构造 ValidationError 失败")
    except ValidationError as exc:
        validation_error = exc

    _req = httpx.Request("POST", "http://example.invalid")

    cases: list[tuple[str, Exception, bool]] = [
        ("模型输出不合 schema（ValidationError）", validation_error, True),
        ("网络连接失败（ConnectionError）", ConnectionError(), True),
        (
            "限流（429）",
            openai.RateLimitError("x", response=httpx.Response(429, request=_req), body=None),
            True,
        ),
        (
            "服务端错误（500）",
            openai.InternalServerError("x", response=httpx.Response(500, request=_req), body=None),
            True,
        ),
        (
            "API key 无效（401）",
            openai.AuthenticationError("x", response=httpx.Response(401, request=_req), body=None),
            False,
        ),
        (
            "模型名写错（404）",
            openai.NotFoundError("x", response=httpx.Response(404, request=_req), body=None),
            False,
        ),
        ("代码 bug（TypeError）", TypeError(), False),
    ]

    failed = 0
    for label, exc, expected in cases:
        actual = retry_on_llm_error(exc)
        ok = actual == expected
        failed += 0 if ok else 1
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        verdict = "重试" if actual else "不重试"
        console.print(f"  {mark} {label:<34} → {verdict}")

    rule("结论")
    if failed:
        console.print(f"[bold red]{failed} 个用例不符合预期[/bold red]")
    else:
        console.print("[bold green]全部符合预期[/bold green]")
        console.print(
            "\n  [dim]对比一下：如果直接用 RetryPolicy(max_attempts=3)，"
            "第一行会是「不重试」。[/dim]"
        )
