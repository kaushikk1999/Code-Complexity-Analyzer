from analyzer import analyze_code
from optimization.planner import build_optimization_plan
from scoring import calculate_optimization_score


def test_already_optimal_two_sum_does_not_show_unproven_reference_code():
    code = '''
def two_sum(nums, target):
    """Return indices of two numbers that add to target using a hash map."""
    seen = {}
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return []
'''

    analysis = analyze_code(code)
    score = calculate_optimization_score(analysis)
    plan = build_optimization_plan(analysis, score, entrypoint="two_sum")

    assert plan.optimized_code is None
    assert plan.validation.status == "not_generated"
    assert plan.validation.original_time == analysis.estimated_time
    assert plan.validation.original_space == analysis.estimated_space
