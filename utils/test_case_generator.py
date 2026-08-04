"""Generate local benchmark/test cases from detected entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List

from utils.entrypoints import EntrypointDefinition, find_entrypoint_definition

DEFAULT_BENCHMARK_CASE_COUNT = 20


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


def _annotation_for_arg(definition: EntrypointDefinition, arg: str) -> str:
    return (definition.annotations.get(arg, "") or "").lower()


def _arg_expects_string(definition: EntrypointDefinition, arg: str) -> bool:
    lowered = arg.lower()
    annotation = _annotation_for_arg(definition, arg)
    return annotation == "str" or any(token in lowered for token in ("password", "text", "string", "word"))


def _arg_expects_sequence(definition: EntrypointDefinition, arg: str) -> bool:
    annotation = _annotation_for_arg(definition, arg)
    lowered = arg.lower()
    return annotation in {"list", "sequence"} or any(token in lowered for token in ("nums", "arr", "array", "list"))


def build_benchmark_batch_input(cases: List[GeneratedTestCase]) -> str:
    if not cases:
        return ""

    payload_cases = []
    for case in cases:
        try:
            payload = json.loads(case.benchmark_input)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not any(key in payload for key in ("args", "kwargs", "stdin")):
            continue
        named_payload = dict(payload)
        named_payload.setdefault("name", case.name)
        payload_cases.append(named_payload)

    if not payload_cases:
        return cases[0].benchmark_input
    if len(payload_cases) == 1:
        return json.dumps(payload_cases[0], ensure_ascii=False, indent=2)
    return json.dumps({"cases": payload_cases}, ensure_ascii=False, indent=2)


def _case_payload(case: GeneratedTestCase) -> dict | None:
    try:
        payload = json.loads(case.benchmark_input)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not any(key in payload for key in ("args", "kwargs", "stdin")):
        return None
    named_payload = dict(payload)
    named_payload.setdefault("name", case.name)
    return named_payload


def benchmark_payload_case_count(input_text: str) -> int:
    text = (input_text or "").strip()
    if not text:
        return 0
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return 1
    if isinstance(value, dict) and isinstance(value.get("cases"), list):
        return len(value["cases"])
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return len(value)
    return 1


def merge_generated_and_custom_benchmark_input(cases: List[GeneratedTestCase], custom_input: str = "") -> str:
    payload_cases = [payload for case in cases for payload in [_case_payload(case)] if payload is not None]
    custom_text = (custom_input or "").strip()
    if custom_text:
        try:
            custom_value = json.loads(custom_text)
        except json.JSONDecodeError:
            custom_value = custom_text
        if isinstance(custom_value, dict) and isinstance(custom_value.get("cases"), list):
            payload_cases.extend(item for item in custom_value["cases"] if isinstance(item, dict))
        elif isinstance(custom_value, list) and custom_value and all(isinstance(item, dict) for item in custom_value):
            payload_cases.extend(custom_value)
        elif isinstance(custom_value, dict):
            payload_cases.append(custom_value)
        else:
            payload_cases.append({"args": [custom_value], "name": "Custom input"})
    if not payload_cases:
        return ""
    return json.dumps({"cases": payload_cases}, ensure_ascii=False, indent=2)


def _number_for_arg(arg: str, index: int) -> Any:
    lowered = arg.lower()
    if "dict" in lowered:
        return ["a", "aa", "aaa", "b"][: max(1, (index % 4) + 1)]
    if any(token in lowered for token in ("target", "sum", "val", "k", "n", "x")):
        return index
    return index


def _list_case(arg: str, index: int) -> Any:
    variants = [
        [],
        [1],
        [1, 2, 3],
        [3, 2, 1],
        [1, 2, 2, 3],
        [-3, -1, 0, 2],
        list(range(index % 12)),
        [index, index + 1, index + 2],
    ]
    lowered = arg.lower()
    if any(token in lowered for token in ("s", "text", "string", "word")) and "num" not in lowered:
        return "".join(chr(97 + ((index + offset) % 26)) for offset in range(max(1, index % 10)))
    return variants[index % len(variants)]


def _fallback_cases_for_args(
    args: List[str],
    target_count: int,
    definition: EntrypointDefinition | None = None,
) -> List[GeneratedTestCase]:
    cases: List[GeneratedTestCase] = []
    for index in range(target_count):
        kwargs: dict[str, Any] = {}
        for arg_index, arg in enumerate(args):
            if arg_index == 0:
                if definition and _arg_expects_string(definition, arg) and not _arg_expects_sequence(definition, arg):
                    kwargs[arg] = _string_case(arg, index)
                else:
                    kwargs[arg] = _list_case(arg, index)
            else:
                kwargs[arg] = _number_for_arg(arg, index + arg_index)
        cases.append(
            GeneratedTestCase(
                name=f"Generated coverage case {index + 1}",
                benchmark_input=_json_kwargs(**kwargs),
                reason="Deterministic local fallback case for mandatory benchmark coverage.",
            )
        )
    return cases


def _string_case(arg: str, index: int) -> str:
    lowered = arg.lower()
    if "password" in lowered:
        variants = ["", "a", "abcdef", "Abcdef1", "Abcdef1!", "P@ssw0rd123"]
    else:
        variants = ["", "a", "abc", "leetcode", "catsandog", "hello world"]
    return variants[index % len(variants)]


def _pad_cases(
    cases: List[GeneratedTestCase],
    args: List[str],
    target_count: int,
    definition: EntrypointDefinition | None = None,
) -> List[GeneratedTestCase]:
    if len(cases) >= target_count:
        return cases[:target_count]
    padded = list(cases)
    fallback = _fallback_cases_for_args(args, target_count, definition)
    for case in fallback:
        if len(padded) >= target_count:
            break
        padded.append(case)
    return padded[:target_count]


def generate_test_cases(
    code: str,
    entrypoint: str,
    definitions: List[EntrypointDefinition],
    target_count: int = DEFAULT_BENCHMARK_CASE_COUNT,
) -> List[GeneratedTestCase]:
    definition = find_entrypoint_definition(definitions, entrypoint)
    if not definition:
        return []

    name = _function_name(entrypoint)
    args = _args_for_entrypoint(definition)

    if name == "wordbreak" and args[:2] == ["s", "wordDict"]:
        return _pad_cases([
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
        ], args, target_count, definition)

    if len(args) == 1:
        arg = args[0]
        if _arg_expects_string(definition, arg) and not _arg_expects_sequence(definition, arg):
            return _pad_cases([
                GeneratedTestCase("Empty string", _json_kwargs(**{arg: ""}), reason="Checks empty text input."),
                GeneratedTestCase("Single character", _json_kwargs(**{arg: "a"}), reason="Checks smallest non-empty text input."),
                GeneratedTestCase("Short lowercase text", _json_kwargs(**{arg: "abcdef"}), reason="Checks typical text input."),
                GeneratedTestCase("Mixed characters", _json_kwargs(**{arg: "Abcdef1!"}), reason="Checks mixed text characters."),
                GeneratedTestCase("Longer text", _json_kwargs(**{arg: "P@ssw0rd123"}), reason="Checks a longer string input."),
            ], args, target_count, definition)
        return _pad_cases([
            GeneratedTestCase("Empty list", _json_kwargs(**{arg: []}), reason="Checks empty input."),
            GeneratedTestCase("Single item", _json_kwargs(**{arg: [1]}), reason="Checks smallest non-empty input."),
            GeneratedTestCase("Small sorted list", _json_kwargs(**{arg: [1, 2, 3]}), reason="Checks sorted input."),
            GeneratedTestCase("Duplicates", _json_kwargs(**{arg: [1, 2, 2, 3]}), reason="Checks duplicate values."),
            GeneratedTestCase("Negative values", _json_kwargs(**{arg: [-3, -1, 0, 2]}), reason="Checks negative values."),
        ], args, target_count, definition)

    if len(args) == 2:
        first, second = args
        return _pad_cases([
            GeneratedTestCase("Small positive case", _json_kwargs(**{first: [1, 2, 3], second: 3})),
            GeneratedTestCase("Empty first input", _json_kwargs(**{first: [], second: 0})),
            GeneratedTestCase("Single item", _json_kwargs(**{first: [1], second: 1})),
            GeneratedTestCase("Duplicates", _json_kwargs(**{first: [2, 2, 3], second: 4})),
            GeneratedTestCase("Negative values", _json_kwargs(**{first: [-1, 0, 1], second: 0})),
        ], args, target_count, definition)

    return _pad_cases([
        GeneratedTestCase(
            name="Default generated case",
            benchmark_input=json.dumps({"args": []}),
            reason="Could not infer a specific input shape.",
        )
    ], args, target_count, definition)
