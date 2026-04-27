"""Example snippets for the Streamlit app."""

from __future__ import annotations

from typing import Dict

EXAMPLES: Dict[str, Dict[str, str]] = {
    "Two Sum - Brute Force": {
        "entrypoint": "two_sum",
        "input": '{"args": [[2, 7, 11, 15, 21, 30, 42, 55], 57]}',
        "code": '''def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
    },
    "Duplicate Detector - List Membership": {
        "entrypoint": "find_duplicates",
        "input": '{"args": [[1, 2, 3, 2, 4, 5, 1, 6, 3]]}',
        "code": '''def find_duplicates(values):
    seen = []
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.append(value)
    return duplicates
''',
    },
    "Sorted Scores": {
        "entrypoint": "top_scores",
        "input": '{"args": [[88, 91, 73, 99, 84, 91, 76], 3]}',
        "code": '''def top_scores(scores, k):
    ordered = sorted(scores, reverse=True)
    return ordered[:k]
''',
    },
    "Recursive Fibonacci": {
        "entrypoint": "fib",
        "input": '{"args": [10]}',
        "code": '''def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
''',
    },
    "Binary Search": {
        "entrypoint": "binary_search",
        "input": '{"args": [[1, 3, 5, 7, 9, 11], 7]}',
        "code": '''def binary_search(values, target):
    left, right = 0, len(values) - 1
    while left <= right:
        mid = (left + right) // 2
        if values[mid] == target:
            return mid
        if values[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
    },
    "Sliding Window Max Sum": {
        "entrypoint": "max_window_sum",
        "input": '{"args": [[2, 1, 5, 1, 3, 2], 3]}',
        "code": '''def max_window_sum(values, k):
    window_sum = sum(values[:k])
    best = window_sum
    left = 0
    for right in range(k, len(values)):
        window_sum += values[right]
        window_sum -= values[left]
        left += 1
        best = max(best, window_sum)
    return best
''',
    },
    "Top K With Heap": {
        "entrypoint": "top_k",
        "input": '{"args": [[5, 1, 9, 3, 7, 8], 3]}',
        "code": '''from heapq import heappush, heappop

def top_k(values, k):
    heap = []
    for value in values:
        heappush(heap, value)
        if len(heap) > k:
            heappop(heap)
    return sorted(heap, reverse=True)
''',
    },
    "Memoized Fibonacci": {
        "entrypoint": "fib",
        "input": '{"args": [20]}',
        "code": '''from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
''',
    },
    "Graph BFS": {
        "entrypoint": "bfs",
        "input": '{"args": [{"0": ["1", "2"], "1": ["3"], "2": [], "3": []}, "0"]}',
        "code": '''from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
''',
    },
}


def get_example(name: str) -> Dict[str, str]:
    return EXAMPLES[name]
