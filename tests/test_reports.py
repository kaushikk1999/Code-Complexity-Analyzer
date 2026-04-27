from analyzer import analyze_code
from optimization import build_optimization_plan
from scoring import calculate_optimization_score
from utils.report_export import build_html_report, build_linkedin_summary, build_markdown_report


def test_reports_include_state_of_the_art_sections():
    analysis = analyze_code(
        """
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
"""
    )
    score = calculate_optimization_score(analysis)
    plan = build_optimization_plan(analysis, score)
    markdown = build_markdown_report(analysis, score, plan, None)
    html = build_html_report(analysis, score, plan, None)
    linkedin = build_linkedin_summary(analysis, score, None)
    assert "Algorithm Pattern Recognition" in markdown
    assert "Line-Level Findings" in markdown
    assert "Verified Code Generation" in markdown
    assert "Memory trade-off" in markdown
    assert "<html" in html
    assert "Verified Code Generation" in html
    assert "optimization score" in linkedin.lower()
