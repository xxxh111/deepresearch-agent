"""验证 RetryPolicy 的真实行为 —— 以及一个会让它静默失效的坑。

    python learn/02b_retry_policy.py

四个实验构成一条完整的推理链：

    实验一  ValueError        + RetryPolicy → 不重试   ← RetryPolicy 好像没用？
    实验二  ConnectionError   + RetryPolicy → 重试     ← 不对，是异常类型的问题
    实验三  ValidationError   + RetryPolicy → 不重试   ← 而这是我们的真实场景！
    实验四  ValidationError   + 自定义 retry_on → 重试 ← 修复验证

实验三是最重要的一步：Pydantic 的 ValidationError 继承自 ValueError，
所以它落进了 LangGraph 的「代码写错了，不重试」名单里。
"""

import sys
import time
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.console  # noqa: E402,F401
from utils.console import console, rule  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import RetryPolicy  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402


class State(TypedDict, total=False):
    done: bool


class TinyPlan(BaseModel):
    """故意把下界设成 3，方便制造验证失败。"""

    sub_questions: list[str] = Field(min_length=3)


def make_node(exc_factory, counter: list, succeed_after: int):
    """造一个「前 succeed_after 次抛异常，之后成功」的节点。

    exc_factory() 每次调用生成一个新异常 —— 异常对象不能复用实例。
    """

    def node(state: State) -> dict:
        counter.append(time.monotonic())
        attempt = len(counter)
        if attempt <= succeed_after:
            raise exc_factory(attempt)
        return {"done": True}

    return node


def build(exc_factory, counter: list, succeed_after: int, retry_on=None):
    builder = StateGraph(State)
    policy = RetryPolicy(max_attempts=3, **({"retry_on": retry_on} if retry_on else {}))
    builder.add_node("flaky", make_node(exc_factory, counter, succeed_after), retry_policy=policy)
    builder.add_edge(START, "flaky")
    builder.add_edge("flaky", END)
    return builder.compile()


def run_case(title: str, exc_factory, succeed_after: int, retry_on=None) -> int:
    """跑一个用例，返回节点被调用的次数。"""
    rule(title)
    counter: list = []
    graph = build(exc_factory, counter, succeed_after, retry_on)
    try:
        graph.invoke({})
        outcome = "[green]图正常结束[/green]"
    except Exception as exc:
        outcome = f"[red]抛出 {type(exc).__name__}[/red]"

    console.print(f"  节点调用次数： [bold]{len(counter)}[/bold]   结果：{outcome}")
    return len(counter)


def experiment_1():
    run_case(
        "实验一：ValueError + RetryPolicy(max_attempts=3)",
        lambda i: ValueError(f"模拟失败 #{i}"),
        succeed_after=2,
    )
    console.print(
        "  [yellow]按理说 max_attempts=3 应该试 3 次，实际只试了 1 次。[/yellow]"
    )
    console.print("  [dim]→ RetryPolicy 好像坏了？继续看[/dim]")


def experiment_2():
    calls = run_case(
        "实验二：ConnectionError + RetryPolicy(max_attempts=3)",
        lambda i: ConnectionError(f"模拟网络失败 #{i}"),
        succeed_after=2,
    )
    if calls >= 3:
        console.print(
            "\n  [green]重试成功了，而且第 3 次救回来了 —— 图没崩。[/green]"
        )
        console.print(
            "  [cyan]→ 所以 RetryPolicy 本身是好的。问题出在**异常类型**上。[/cyan]"
        )
    else:
        console.print("  [yellow]没重试？预期之外，需要重新排查[/yellow]")


def experiment_3():
    """这是我们的真实场景：模型输出不合 schema。"""

    def make_validation_error(i: int):
        try:
            TinyPlan(sub_questions=["只有一条"])
        except ValidationError as exc:
            return exc
        raise AssertionError("构造 ValidationError 失败")

    calls = run_case(
        "实验三：ValidationError + RetryPolicy(max_attempts=3)  ← 我们真正要重试的场景",
        make_validation_error,
        succeed_after=2,
    )

    console.print()
    console.print("  [bold red]坐实了：ValidationError 也不重试。[/bold red]")
    console.print(
        "  原因在 default_retry_on 的源码里 —— 它把 ValueError 列入「不重试」名单，\n"
        "  而 pydantic.ValidationError 恰好是 ValueError 的子类："
    )
    console.print("     ValidationError → ValueError → Exception")
    console.print(
        "\n  [cyan]LangGraph 的原意是：ValueError/TypeError 通常表示**你的代码写错了**，"
        "重试没意义。\n"
        "  但它没考虑到 Pydantic 的 ValidationError 语义完全不同 ——"
        "它表示**模型输出不符合 schema**，\n"
        "  而模型下一轮很可能就对了，这正是最值得重试的情况。[/cyan]"
    )
    console.print(
        "\n  [bold yellow]结论：给 planner 挂 RetryPolicy(max_attempts=3) "
        "在当前配置下等于没挂。[/bold yellow]"
    )


def experiment_4():
    """修复：自定义 retry_on，把 ValidationError 加回重试名单。"""

    from langgraph.types import default_retry_on

    def retry_on_validation(exc: Exception) -> bool:
        # 模型输出不合 schema —— 重试有意义
        if isinstance(exc, ValidationError):
            return True
        # 其余一律沿用 LangGraph 的默认判断
        return default_retry_on(exc)

    def make_validation_error(i: int):
        try:
            TinyPlan(sub_questions=["只有一条"])
        except ValidationError as exc:
            return exc
        raise AssertionError

    calls = run_case(
        "实验四：ValidationError + 自定义 retry_on → 应该重试",
        make_validation_error,
        succeed_after=2,
        retry_on=retry_on_validation,
    )
    if calls >= 3:
        console.print("\n  [green]修复生效：重试了，图正常结束。[/green]")
    else:
        console.print("\n  [red]仍然没重试 —— 需要重新排查[/red]")


def main():
    console.print("\n[bold]验证 RetryPolicy 的真实行为[/bold]")
    console.print("[dim]每次重试有指数退避，脚本大约需要 10 秒[/dim]")

    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()

    rule("你要做的事")
    console.print(
        "  在 graph/build.py 里给 RetryPolicy 补上 retry_on，把 ValidationError\n"
        "  加回重试名单（照抄实验四的 retry_on_validation，或自己写一版更清晰的）。\n"
        "\n"
        "  顺带想两个问题：\n"
        "    1. 这个 retry_on 以后 M3/M7 的节点也要用吗？如果是，应该放哪？\n"
        "    2. 除了 ValidationError，还有哪些异常你觉得值得重试？\n"
        "       （提示：DeepSeek 限流时会抛什么？）\n"
    )


if __name__ == "__main__":
    main()
