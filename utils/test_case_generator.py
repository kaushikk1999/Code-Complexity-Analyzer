"""Generate local benchmark/test cases from detected entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List

from utils.entrypoints import EntrypointDefinition, find_entrypoint_definition


@dataclass
class GeneratedTestCase:
    name: str
    benchmark_input: str
    expected_output: str = ""
    reason: str = ""


def _function_name(entrypoint: str) -> str:
    return (entrypoint or "").split(".")[-1].lower()


def _args_for_entrypoint(definition: EntrypointDefinition) -> List[str]:
    return definition.benchmark_args


def _json_kwargs(**kwargs: Any) -> str:
    return json.dumps({"kwargs": kwargs}, ensure_ascii=False)


def generate_test_cases(
    code: str,
    entrypoint: str,
    definitions: List[EntrypointDefinition],
) -> List[GeneratedTestCase]:
    definition = find_entrypoint_definition(definitions, entrypoint)
    if not definition:
        return []

    name = _function_name(entrypoint)
    args = _args_for_entrypoint(definition)

    if name == "wordbreak" and args[:2] == ["s", "wordDict"]:
        return [
            GeneratedTestCase(
                name="Basic positive segmentation",
                benchmark_input=_json_kwargs(s="leetcode", wordDict=["leet", "code"]),
                expected_output="True",
                reason="Checks the canonical positive case.",
            ),
            GeneratedTestCase(
                name="Basic negative segmentation",
                benchmark_input=_json_kwargs(s="catsandog", wordDict=["cats", "dog", "sand", "and", "cat"]),
                expected_output="False",
                reason="Checks that partial matches do not force a false positive.",
            ),
            GeneratedTestCase(
                name="Repeated word reuse",
                benchmark_input=_json_kwargs(s="applepenapple", wordDict=["apple", "pen"]),
                expected_output="True",
                reason="Checks whether dictionary words can be reused.",
            ),
            GeneratedTestCase(
                name="Empty string",
                benchmark_input=_json_kwargs(s="", wordDict=["a", "abc"]),
                expected_output="True",
                reason="An empty string should be segmentable by definition.",
            ),
            GeneratedTestCase(
                name="Overlapping prefixes",
                benchmark_input=_json_kwargs(s="aaaaaaa", wordDict=["aaaa", "aaa"]),
                expected_output="True",
                reason="Checks overlapping word lengths and DP transitions.",
            ),
        ]

    if len(args) == 1:
        arg = args[0]
        return [
            GeneratedTestCase("Empty list", _json_kwargs(**{arg: []}), reason="Checks empty input."),
            GeneratedTestCase("Single item", _json_kwargs(**{arg: [1]}), reason="Checks smallest non-empty input."),
            GeneratedTestCase("Small sorted list", _json_kwargs(**{arg: [1, 2, 3]}), reason="Checks sorted input."),
            GeneratedTestCase("Duplicates", _json_kwargs(**{arg: [1, 2, 2, 3]}), reason="Checks duplicate values."),
            GeneratedTestCase("Negative values", _json_kwargs(**{arg: [-3, -1, 0, 2]}), reason="Checks negative values."),
        ]

    if len(args) == 2:
        first, second = args
        return [
            GeneratedTestCase("Small positive case", _json_kwargs(**{first: [1, 2, 3], second: 3})),
            GeneratedTestCase("Empty first input", _json_kwargs(**{first: [], second: 0})),
            GeneratedTestCase("Single item", _json_kwargs(**{first: [1], second: 1})),
            GeneratedTestCase("Duplicates", _json_kwargs(**{first: [2, 2, 3], second: 4})),
            GeneratedTestCase("Negative values", _json_kwargs(**{first: [-1, 0, 1], second: 0})),
        ]

    return [
        GeneratedTestCase(
            name="Default generated case",
            benchmark_input=json.dumps({"args": []}),
            reason="Could not infer a specific input shape.",
        )
    ]
