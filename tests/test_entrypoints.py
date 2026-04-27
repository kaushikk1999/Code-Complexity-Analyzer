from pathlib import Path

from utils.entrypoints import choose_entrypoint, discover_entrypoints

BINARY_SEARCH_CODE = """
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


def test_stale_default_entrypoint_is_replaced_for_single_detected_function():
    assert choose_entrypoint(BINARY_SEARCH_CODE, "two_sum") == "binary_search"


def test_multiple_functions_choose_preferred_default_then_keep_valid_selection():
    code = """
def helper(value):
    return value + 1

def solve(value):
    return helper(value)

def later(value):
    return value
"""
    assert choose_entrypoint(code, "missing") == "solve"
    assert choose_entrypoint(code, "later") == "later"


def test_class_methods_are_exposed_as_qualified_entrypoints():
    code = """
class Solution:
    def search(self, nums, target):
        return -1
"""
    definitions = discover_entrypoints(code)
    assert [definition.callable_name for definition in definitions] == ["Solution.search"]
    assert choose_entrypoint(code, "search") == "Solution.search"


def test_app_uses_auto_detected_entrypoint_selectbox():
    app_source = Path("app.py").read_text(encoding="utf-8")
    assert "choose_entrypoint" in app_source
    assert 'st.selectbox(\n            "Entrypoint function"' in app_source
    assert 'st.text_input(\n            "Entrypoint function"' in app_source
