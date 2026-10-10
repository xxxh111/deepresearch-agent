"""引用校验的测试。

重点覆盖一个**曾经真实存在的假阴性**：
第一版校验器拿整份报告（含参考文献节）去扫引用编号，
而参考文献节的格式就是 `[n] 标题. URL` —— 所有编号必然都出现，
于是 unused 永远是空集。校验器跑了、通过了、什么也没查出来。
"""

from schemas.research import Evidence
from utils.report import body_only, collect_citations, validate_citations


def _evidence(index: int, url: str) -> Evidence:
    return Evidence(
        claim=f"结论 {index}",
        quote="原文片段",
        source_url=url,
        source_title=f"来源 {index}",
        sub_question="某个子问题",
        citation_index=index,
    )


def test_collect_citations_finds_all():
    text = "这里引用了 [1] 和 [12]，还有并列的 [3][4]。"
    assert collect_citations(text) == {1, 12, 3, 4}


def test_body_only_strips_references():
    report = "正文引用 [1]。\n\n## 参考文献\n[1] 甲. https://a.com\n[2] 乙. https://b.com\n"
    body = body_only(report)
    assert "[1]" in body
    assert "[2]" not in body, "参考文献节必须被排除，否则校验必然通过"


def test_detects_fabricated_citation():
    """报告引用了证据表里没有的编号 —— 这是防幻觉的核心检查。"""
    report = "结论 A [1]。结论 B [7]。\n\n## 参考文献\n[1] 甲. https://a.com\n"
    evidence = [_evidence(1, "https://a.com")]

    result = validate_citations(report, evidence)
    assert result["missing"] == [7]
    assert result["used"] == {1, 7}


def test_detects_unused_citation_listed_only_in_references():
    """回归测试：参考文献节里列了、但正文一次都没引用的编号必须被查出来。

    这正是当初假阴性的场景 —— 如果校验器把参考文献节也算作「引用」，
    这个断言会失败。
    """
    report = (
        "正文只引用了 [1]。\n\n"
        "## 参考文献\n"
        "[1] 甲. https://a.com\n"
        "[2] 乙. https://b.com\n"
    )
    evidence = [_evidence(1, "https://a.com"), _evidence(2, "https://b.com")]

    result = validate_citations(report, evidence)
    assert result["unused"] == [2], "参考文献节里的编号不算「被引用」"
    assert result["missing"] == []


def test_clean_report_passes():
    report = "结论 A [1]。结论 B [2]。\n\n## 参考文献\n[1] 甲. https://a.com\n[2] 乙. https://b.com\n"
    evidence = [_evidence(1, "https://a.com"), _evidence(2, "https://b.com")]

    result = validate_citations(report, evidence)
    assert result["missing"] == []
    assert result["unused"] == []
