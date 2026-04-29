"""Entrypoint discovery and benchmark input helpers."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PREFERRED_ENTRYPOINT_NAMES = ("main", "solve", "solution")


@dataclass(frozen=True)
class EntrypointDefinition:
    name: str
    qualified_name: str
    args: List[str]
    defaults_count: int = 0
    class_name: str = ""
    has_varargs: bool = False
    has_varkw: bool = False
    keyword_only_args: List[str] = field(default_factory=list)
    required_keyword_only_args: List[str] = field(default_factory=list)

    @property
    def is_method(self) -> bool:
        return bool(self.class_name)

    @property
    def callable_name(self) -> str:
        return self.qualified_name if self.is_method else self.name

    @property
    def benchmark_args(self) -> List[str]:
        if self.is_method and self.args and self.args[0] in {"self", "cls"}:
            return self.args[1:]
        return list(self.args)

    @property
    def required_positional_count(self) -> int:
        defaults = min(self.defaults_count, len(self.benchmark_args))
        return max(0, len(self.benchmark_args) - defaults)


def _definition_from_function(node: ast.FunctionDef, class_name: str = "") -> Optional[EntrypointDefinition]:
    if node.name.startswith("__"):
        return None
    required_keyword_only = [
        arg.arg for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is None
    ]
    return EntrypointDefinition(
        name=node.name,
        qualified_name=f"{class_name}.{node.name}" if class_name else node.name,
        args=[arg.arg for arg in node.args.args],
        defaults_count=len(node.args.defaults),
        class_name=class_name,
        has_varargs=node.args.vararg is not None,
        has_varkw=node.args.kwarg is not None,
        keyword_only_args=[arg.arg for arg in node.args.kwonlyargs],
        required_keyword_only_args=required_keyword_only,
    )


def discover_entrypoints(code: str) -> List[EntrypointDefinition]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    definitions: List[EntrypointDefinition] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            definition = _definition_from_function(node)
            if definition:
                definitions.append(definition)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    definition = _definition_from_function(child, node.name)
                    if definition:
                        definitions.append(definition)
    return definitions


def find_entrypoint_definition(
    definitions: List[EntrypointDefinition],
    entrypoint: str,
) -> Optional[EntrypointDefinition]:
    entrypoint = (entrypoint or "").strip()
    if not entrypoint:
        return None
    matches = [
        definition
        for definition in definitions
        if entrypoint in {definition.name, definition.qualified_name, definition.callable_name}
    ]
    if len(matches) == 1:
        return matches[0]
    for definition in matches:
        if entrypoint == definition.callable_name:
            return definition
    return None


def choose_entrypoint(code: str, current_entrypoint: str = "") -> str:
    definitions = discover_entrypoints(code)
    current = (current_entrypoint or "").strip()
    if not definitions:
        return current
    current_definition = find_entrypoint_definition(definitions, current)
    if current_definition:
        return current_definition.callable_name
    if len(definitions) == 1:
        return definitions[0].callable_name
    for preferred in PREFERRED_ENTRYPOINT_NAMES:
        preferred_definition = find_entrypoint_definition(definitions, preferred)
        if preferred_definition:
            return preferred_definition.callable_name
    top_level = [definition for definition in definitions if not definition.is_method]
    return (top_level[0] if top_level else definitions[0]).callable_name


def format_available_entrypoints(definitions: List[EntrypointDefinition]) -> str:
    if not definitions:
        return "none"
    return ", ".join(definition.callable_name for definition in definitions)


def benchmark_input_hint(definition: EntrypointDefinition) -> str:
    required = definition.benchmark_args[: definition.required_positional_count]
    sample = {"args": required, "kwargs": {name: name for name in definition.required_keyword_only_args}}
    hint = f"Expected input shape for `{definition.callable_name}`: {json.dumps(sample)}."
    optional = definition.benchmark_args[definition.required_positional_count :]
    if optional:
        hint += f" Optional positional args with defaults: {', '.join(optional)}."
    hint += " Assignment-style input is also supported, for example `arr = [1, 2, 3]`."
    return hint


def missing_entrypoint_message(entrypoint: str, definitions: List[EntrypointDefinition]) -> str:
    if definitions:
        suggestion = "Choose one of the available entrypoints in the sidebar."
        if len(definitions) == 1:
            suggestion = f"Use `{definitions[0].callable_name}` as the entrypoint."
        return (
            f"Entrypoint `{entrypoint}` was not found or is not callable. "
            f"Available callable entrypoints: {format_available_entrypoints(definitions)}. {suggestion}"
        )
    return (
        f"Entrypoint `{entrypoint}` was not found or is not callable. "
        "No function or class method entrypoints were detected; leave the entrypoint blank only when top-level "
        "script benchmarking is enabled."
    )


def validate_call_arguments(
    definition: EntrypointDefinition,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> str:
    positional_names = definition.benchmark_args
    supplied_positionally = set(positional_names[: len(args)])
    duplicate_kwargs = sorted(name for name in kwargs if name in supplied_positionally)
    if duplicate_kwargs:
        return (
            f"Benchmark input passes {', '.join(duplicate_kwargs)} both positionally and by keyword. "
            f"{benchmark_input_hint(definition)}"
        )

    required_positional = positional_names[: definition.required_positional_count]
    missing_positional = [
        name for index, name in enumerate(required_positional) if index >= len(args) and name not in kwargs
    ]
    missing_keyword_only = [name for name in definition.required_keyword_only_args if name not in kwargs]
    missing = missing_positional + missing_keyword_only
    if missing:
        return (
            f"Benchmark input is missing required argument(s): {', '.join(missing)}. "
            f"{benchmark_input_hint(definition)}"
        )

    if not definition.has_varargs and len(args) > len(positional_names):
        return (
            f"Benchmark input provides {len(args)} positional args, but `{definition.callable_name}` accepts "
            f"at most {len(positional_names)}. {benchmark_input_hint(definition)}"
        )

    accepted_kwargs = set(positional_names) | set(definition.keyword_only_args)
    unknown_kwargs = sorted(name for name in kwargs if name not in accepted_kwargs)
    if unknown_kwargs and not definition.has_varkw:
        return (
            f"Benchmark input has unexpected keyword argument(s): {', '.join(unknown_kwargs)}. "
            f"{benchmark_input_hint(definition)}"
        )
    return ""


def normalize_leetcode_style_assignments(text: str) -> str:
    """Convert LeetCode-style copied input into Python assignment input.

    Example:
        s =
        "leetcode"
        wordDict =
        ["leet", "code"]

    becomes:
        s = "leetcode"
        wordDict = ["leet", "code"]
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    normalized = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.endswith("=") and index + 1 < len(lines):
            name = line[:-1].strip()
            value = lines[index + 1].strip()
            if name.isidentifier():
                normalized.append(f"{name} = {value}")
                index += 2
                continue

        normalized.append(line)
        index += 1

    return "\n".join(normalized)


def literal_assignment_namespace(text: str) -> Dict[str, Any]:
    text = normalize_leetcode_style_assignments(text or "")
    tree = ast.parse(text or "", mode="exec")
    namespace: Dict[str, Any] = {}
    for node in tree.body:
        value_node = None
        targets: List[ast.AST] = []
        if isinstance(node, ast.Assign):
            value_node = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value_node = node.value
            targets = [node.target]
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is None:
            continue
        else:
            raise ValueError("assignment-style benchmark input can only contain literal variable assignments.")
        if value_node is None:
            raise ValueError("assignment-style benchmark input requires literal values.")
        value = ast.literal_eval(value_node)
        for target in targets:
            if not isinstance(target, ast.Name):
                raise ValueError("assignment-style benchmark input supports simple variable names only.")
            namespace[target.id] = value
    if not namespace:
        raise ValueError("assignment-style benchmark input did not define any variables.")
    return namespace


def _call_matches(node: ast.Call, definition: EntrypointDefinition) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id == definition.name
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == definition.name
    return False


def _literal_or_name_value(node: ast.AST, namespace: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Name) and node.id in namespace:
        return namespace[node.id]
    return ast.literal_eval(node)


def infer_call_from_example(
    code: str,
    definition: EntrypointDefinition,
    namespace: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any], bool]:
    try:
        tree = ast.parse(code or "", mode="exec")
    except SyntaxError:
        return (), {}, False

    module_statements = [
        node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for statement in module_statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call) or not _call_matches(node, definition):
                continue
            try:
                args = tuple(_literal_or_name_value(arg, namespace) for arg in node.args)
                kwargs = {
                    keyword.arg: _literal_or_name_value(keyword.value, namespace)
                    for keyword in node.keywords
                    if keyword.arg
                }
            except Exception:
                continue
            return args, kwargs, True
    return (), {}, False
