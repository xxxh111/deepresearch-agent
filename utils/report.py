"""报告落盘 + 引用校验。

【为什么要校验引用】
提示词里要求 writer「只引用证据表里给出的编号」，但**要求不等于保证**。
模型完全可能写出一个 [7]，而证据表里根本没有 7 号来源 ——
读者点进去发现对不上，整份报告的可信度就没了。

所以生成之后要**机器校验一遍**：把报告里出现的所有 [n] 抽出来，
和证据表里的编号做差集。这是把「模型的自觉」换成「确定性的检查」。

校验结果不阻断流程（报告照常输出），但会：
  · 在终端明确警告
  · 在报告末尾追加一条说明
让问题**可见**，而不是悄悄混过去。
"""

import re
from datetime import datetime
from pathlib import Path

from schemas.research import Evidence
from utils.path_tool import ensure_dir

# 匹配 [1] [23] 这样的引用标记
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# 参考文献节的标题。校验时必须把它排除掉，原因见 body_only() 的说明。
_REFERENCES_HEADING = "## 参考文献"


def collect_citations(text: str) -> set[int]:
    """抽出文本里出现的所有引用编号。"""
    return {int(m) for m in _CITATION_PATTERN.findall(text or "")}


def body_only(report: str) -> str:
    """去掉「参考文献」一节，只留正文。

    ============================ 为什么必须这么做 ============================

    这是校验器第一版的 bug —— 而且是**假阴性**，最难发现的那种。

    第一版拿整份报告去扫引用编号。但参考文献节的格式就是：

        ## 参考文献
        [1] 某论文. https://...
        [2] 某博客. https://...
        [21] ...

    也就是说，**参考文献节里必然出现所有编号**。于是「哪些编号没被正文引用」
    这个检查永远返回空集 —— 校验器看起来每天都在跑、每次都通过，
    实际上一个都没查出来。

    ⚠️ 教训：**写校验逻辑时，先问「什么情况下它会必然通过」。**
       一个永远通过的校验器比没有校验器更糟 —— 它给你虚假的安全感。

    ===========================================================================
    """
    return report.split(_REFERENCES_HEADING, 1)[0]


def validate_citations(report: str, evidence: list[Evidence]) -> dict:
    """校验报告里的引用编号。

    Returns:
        {
          "used":    正文里实际引用的编号集合（不含参考文献节）,
          "missing": 正文引用了、但证据表里没有的编号（编造的）,
          "unused":  证据表里有、但正文一次都没引用的编号（白收集了）,
        }
    """
    available = {e.citation_index for e in evidence if e.citation_index}
    used = collect_citations(body_only(report))

    return {
        "used": used,
        "missing": sorted(used - available),
        "unused": sorted(available - used),
    }


def save_report(topic: str, report: str, thread_id: str) -> Path:
    """把报告写到 reports/{thread_id}_{时间}.md。

    文件名带上 thread_id，是为了能顺着文件名找回那条执行历史
    ——「这份报告是哪次运行产出的」这个问题应该一眼能答。
    """
    report_dir = ensure_dir("reports")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = Path(report_dir) / f"{thread_id}_{stamp}.md"

    header = (
        f"# {topic}\n\n"
        f"> 由 DeepResearch Agent 生成 · "
        f"{datetime.now():%Y-%m-%d %H:%M} · thread_id: `{thread_id}`\n\n"
        f"---\n\n"
    )
    path.write_text(header + report, encoding="utf-8")
    return path


def append_warning(report: str, message: str) -> str:
    """把校验警告追加到报告末尾。"""
    return f"{report}\n\n---\n\n> ⚠️ **自动校验提示**：{message}\n"


if __name__ == "__main__":
    from utils.console import console, rule

    rule("引用抽取消验")

    # 关键：参考文献节里列了 [3]，但正文一次都没引用它。
    # 这正是当初校验器假阴性的场景 —— 如果把参考文献节也算进「引用」，
    # unused 就会变成空集，看起来一切正常。
    sample_report = """
## 摘要
多智能体架构在长程任务上表现更好 [1][2]。但也有研究持保留态度 [7]。

## 参考文献
[1] 某论文. https://a.com
[2] 某博客. https://b.com
[3] 某没人引用的来源. https://c.com
"""
    sample_evidence = [
        Evidence(
            claim="c1", quote="q1", source_url="https://a.com",
            source_title="某论文", sub_question="sq", citation_index=1,
        ),
        Evidence(
            claim="c2", quote="q2", source_url="https://b.com",
            source_title="某博客", sub_question="sq", citation_index=2,
        ),
        Evidence(
            claim="c3", quote="q3", source_url="https://c.com",
            source_title="没人引用它", sub_question="sq", citation_index=3,
        ),
    ]

    result = validate_citations(sample_report, sample_evidence)
    console.print(f"  报告引用了    : {sorted(result['used'])}")
    console.print(f"  证据表提供    : [1, 2, 3]")
    console.print(f"  [red]编造的编号  : {result['missing']}[/red]  ← [7] 根本不存在")
    console.print(f"  [yellow]没被引用的  : {result['unused']}[/yellow]  ← [3] 白收集了")
