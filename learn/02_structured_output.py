"""M2 教学演示：为什么需要 with_structured_output，以及它到底怎么工作的。

    conda activate langgraph
    python learn/02_structured_output.py

这个脚本刻意按「从坏到好」的顺序演示四种做法，让你亲眼看到前三种怎么崩的。
看完再回去读 graph/nodes.py 里 planner_node 的真实实现。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.console  # noqa: E402,F401  —— 副作用：修复中文编码
from utils.console import console, rule  # noqa: E402

from model.factory import get_chat_model  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

QUESTION = "为什么多智能体架构比单智能体更适合深度研究任务"

# 我们要模型产出的数据结构：把一个问题拆成若干子问题
PROMPT = f"""把下面这个研究问题拆成 3 个可以独立检索的子问题。

研究问题：{QUESTION}
"""


# ============================================================================
# 做法一：纯提示词要 JSON —— 最常见，也最脆
# ============================================================================
def way1_prompt_json():
    rule("做法一：提示词里写「请返回 JSON」")

    llm = get_chat_model("planner")
    resp = llm.invoke(PROMPT + '\n请以 JSON 格式返回，格式：{"sub_questions": ["...", "...", "..."]}')
    text = resp.content

    console.print("[bold]模型实际返回的原始文本：[/bold]")
    console.print(f"[dim]{text}[/dim]\n")

    console.print("[bold]尝试 json.loads()：[/bold]")
    try:
        data = json.loads(text)
        console.print(f"[green]这次成功了：{data}[/green]")
        console.print(
            "[yellow]但是——它这次成功不代表下次成功。模型可能加 ```json 围栏、"
            "在 JSON 前写一句「好的，以下是…」、或者少一个逗号。[/yellow]"
        )
    except json.JSONDecodeError as exc:
        console.print(f"[red]解析失败：{exc}[/red]")
        console.print(
            "[yellow]看到了吗？流程会在这里直接崩。你只能再加正则去剥 ``` 围栏、"
            "再写重试逻辑……越补越脏。[/yellow]"
        )


# ============================================================================
# 做法二：with_structured_output —— 正解
# ============================================================================
class Plan(BaseModel):
    """研究计划。注意每个字段的 description —— 后面会讲它有多重要。"""

    rationale: str = Field(description="一句话说明为什么这样拆解")
    sub_questions: list[str] = Field(description="3 个可独立检索的子问题", min_length=3, max_length=3)


def way2_structured_output():
    rule("做法二：with_structured_output(Plan)")

    llm = get_chat_model("planner")
    structured_llm = llm.with_structured_output(Plan)

    result = structured_llm.invoke(PROMPT)

    console.print("[bold]返回的是什么类型？[/bold]")
    console.print(f"  type(result) = [cyan]{type(result).__name__}[/cyan]")
    console.print("  [dim]→ 不是 str，直接就是 Pydantic 对象，不用 json.loads[/dim]\n")

    console.print("[bold]内容：[/bold]")
    console.print(f"  rationale      = {result.rationale}")
    for i, q in enumerate(result.sub_questions, 1):
        console.print(f"  sub_questions[{i}] = {q}")

    console.print("\n[bold]字段访问有类型保证：[/bold]")
    console.print(f"  result.sub_questions 是 list 吗？ {isinstance(result.sub_questions, list)}")
    console.print("  [dim]→ 下游代码可以放心 .append() / 遍历，不用防御性检查[/dim]")


# ============================================================================
# 做法三：看清 with_structured_output 到底往网络上发了什么
# ============================================================================
def way3_see_the_schema():
    rule("做法三：它到底怎么做到的？看它发给模型的东西")

    schema = Plan.model_json_schema()
    console.print("[bold]Pydantic 类 → JSON Schema：[/bold]")
    console.print_json(data=schema)

    console.print(
        "\n[bold]关键理解：[/bold]\n"
        "  with_structured_output 把你的 Pydantic 类转成上面这个 JSON Schema，\n"
        "  然后**伪装成一个工具（function）注册给模型**。\n"
        "  模型以为自己在「调用一个叫 Plan 的工具」，实际上是按 schema 填参数。\n"
        "  LangChain 再把模型填好的参数反序列化成 Pydantic 对象返回给你。\n"
        "\n"
        "  [cyan]所以它的本质是：借用 tool calling 机制做结构化输出。[/cyan]\n"
        "  这也解释了为什么它的参数叫 method=\"function_calling\"。\n"
    )


# ============================================================================
# 做法四：Field 的 description 不是注释，它就是提示词
# ============================================================================
class PlanVague(BaseModel):
    """故意写得含糊，看看模型会填成什么样。"""

    sub_questions: list[str] = Field(description="子问题")


class PlanPrecise(BaseModel):
    """把要求写清楚。"""

    sub_questions: list[str] = Field(
        description=(
            "3 个子问题。每条都必须是**可以独立拿去搜索引擎检索**的完整问句，"
            "彼此不重叠，合起来能覆盖原问题的全貌。"
            "不要写成「核心概念」这种短语，要写成完整的问句。"
        ),
        min_length=3,
        max_length=3,
    )


def way4_description_matters():
    rule("做法四：description 写得含糊 vs 写清楚")

    llm = get_chat_model("planner")

    console.print("[bold]含糊版 description（「子问题」三个字）：[/bold]")
    try:
        vague = llm.with_structured_output(PlanVague).invoke(PROMPT)
        for q in vague.sub_questions:
            console.print(f"  · {q}")
    except Exception as exc:
        console.print(f"  [red]{type(exc).__name__}: {exc}[/red]")

    console.print("\n[bold]精确版 description：[/bold]")
    precise = llm.with_structured_output(PlanPrecise).invoke(PROMPT)
    for q in precise.sub_questions:
        console.print(f"  · {q}")

    console.print(
        "\n[cyan]同一个模型、同一个问题，只改了 description —— 输出质量差这么多。[/cyan]\n"
        "  因为 description 会**一起发给模型**，它和提示词是一个性质的东西。\n"
        "  很多人把 Field 的 description 当注释随便写，这是浪费了最强的控制手段。"
    )


def main():
    console.print("\n[bold]M2 教学：结构化输出[/bold]")
    console.print("[dim]每一步都会真实调用 DeepSeek，请稍等[/dim]")

    way1_prompt_json()
    way2_structured_output()
    way3_see_the_schema()
    way4_description_matters()

    rule("小结")
    console.print(
        "  1. LLM 输出是不可信文本，json.loads + 正则是脆弱方案\n"
        "  2. with_structured_output 借 tool calling 机制拿到强类型对象\n"
        "  3. 解析失败会**抛异常**，能被重试机制接住 —— 而不是悄悄产出坏数据\n"
        "  4. Field 的 description 就是提示词，要当提示词写\n"
        "  5. 列表必须包在容器模型里（下一节讲为什么）\n"
    )


if __name__ == "__main__":
    main()
