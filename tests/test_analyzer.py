from analyzer import analyze_code


def test_linear_loop_estimate():
    result = analyze_code(
        """
def total(values):
    acc = 0
    for value in values:
        acc += value
    return acc
"""
    )
    assert result.valid
    assert result.estimated_time == "O(n)"
    assert result.metrics["max_loop_depth"] == 1
    assert result.confidence > 0.5


def test_nested_loop_estimate():
    result = analyze_code(
        """
def pairs(values):
    out = []
    for a in values:
        for b in values:
            out.append((a, b))
    return out
"""
    )
    assert result.estimated_time == "O(n^2)"
    assert result.metrics["max_loop_depth"] == 2
    assert any("Nested" in pattern.name for pattern in result.anti_patterns)


def test_recursion_detection():
    result = analyze_code(
        """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"""
    )
    assert result.metrics["recursive_calls"] == 2
    assert result.estimated_time == "O(2^n)"
    assert any("Recursive" in pattern.name for pattern in result.anti_patterns)


def test_sort_and_slicing_detection():
    result = analyze_code(
        """
def top(values):
    ordered = sorted(values)
    return ordered[:3]
"""
    )
    assert result.metrics["sort_calls"] == 1
    assert result.metrics["slicing_count"] == 1
    assert result.estimated_time == "O(n log n)"
    assert result.estimated_space == "O(n)"


def test_comprehension_detection():
    result = analyze_code(
        """
def squares(values):
    return [value * value for value in values if value > 0]
"""
    )
    assert result.metrics["comprehension_count"] == 1
    assert result.estimated_time == "O(n)"
    assert result.estimated_space == "O(n)"


def test_membership_in_loop_detection():
    result = analyze_code(
        """
def find(values):
    out = []
    for value in values:
        if value in [1, 2, 3]:
            out.append(value)
    return out
"""
    )
    assert result.metrics["list_membership_in_loop"] >= 1
    assert any("membership" in pattern.name.lower() for pattern in result.anti_patterns)


def test_parse_error_is_structured():
    result = analyze_code("def broken(:\n    pass")
    assert not result.valid
    assert result.parse_error
    assert "SyntaxError" in result.parse_error


def test_binary_search_pattern_and_call_graph():
    result = analyze_code(
        """
def helper(values):
    return len(values)

def binary_search(values, target):
    left, right = 0, helper(values) - 1
    while left <= right:
        mid = (left + right) // 2
        if values[mid] == target:
            return mid
        if values[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    )
    assert result.call_graph["binary_search"] == ["helper"]
    assert any(pattern.name == "Binary search" for pattern in result.algorithm_patterns)
    assert result.confidence_breakdown["overall"] == result.confidence


def test_recursive_binary_search_estimates_logarithmic_complexity():
    result = analyze_code(
        """
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif target < arr[mid]:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)
"""
    )
    assert result.estimated_time == "O(log n)"
    assert result.estimated_space == "O(log n)"
    assert result.metrics["recursive_binary_search_markers"] == 1
    assert any(pattern.name == "Recursive binary search" for pattern in result.algorithm_patterns)
    assert not any(pattern.severity == "critical" for pattern in result.anti_patterns)


def test_line_findings_for_nested_lookup():
    result = analyze_code(
        """
def contains_any(values, lookup):
    matches = []
    for value in values:
        if value in lookup:
            matches.append(value)
    return matches
"""
    )
    assert result.line_findings
    assert any("membership" in finding.title.lower() for finding in result.line_findings)
