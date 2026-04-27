"""Best-effort execution restrictions for benchmarked snippets.

This is intentionally not described as a secure sandbox. It blocks common
dangerous imports, builtins, and calls so interview-prep snippets can be run
with lower risk inside a Streamlit process.
"""

from __future__ import annotations

import ast
import builtins
from typing import Any, Dict, List, Optional, Set, Tuple

ALLOWED_IMPORT_ROOTS = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "typing",
}

OPTIONAL_IMPORT_ROOTS = {"numpy", "pandas"}

DENIED_IMPORT_ROOTS = {
    "builtins",
    "ctypes",
    "importlib",
    "inspect",
    "marshal",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "threading",
    "urllib",
}

DANGEROUS_BUILTINS = {
    "__import__",
    "breakpoint",
    "compile",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

DANGEROUS_ATTRIBUTES = {
    "accept",
    "bind",
    "chmod",
    "chown",
    "chdir",
    "connect",
    "exec",
    "fork",
    "kill",
    "listen",
    "makedirs",
    "mkdir",
    "open",
    "popen",
    "read_text",
    "recv",
    "remove",
    "rename",
    "replace",
    "request",
    "rmdir",
    "send",
    "spawn",
    "system",
    "unlink",
    "urlopen",
    "write",
    "write_text",
}

SAFE_BUILTIN_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "hash",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}


class SafetyValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._validate_import(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._validate_import(node.module or "", node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_BUILTINS:
            self.violations.append(f"Line {node.lineno}: `{node.func.id}` is not allowed during benchmarking.")
        if isinstance(node.func, ast.Attribute) and node.func.attr in DANGEROUS_ATTRIBUTES:
            self.violations.append(
                f"Line {node.lineno}: attribute call `.{node.func.attr}()` is not allowed during benchmarking."
            )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            if not any(isinstance(child, ast.Break) for child in ast.walk(node)):
                self.violations.append(
                    f"Line {node.lineno}: `while True` without a visible break is blocked during benchmarking."
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == "builtins":
                    self.violations.append(
                        f"Line {node.lineno}: monkey-patching builtins is blocked during benchmarking."
                    )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mult):
            if self._large_multiplier(node.left) or self._large_multiplier(node.right):
                self.violations.append(
                    f"Line {node.lineno}: very large literal allocation is blocked during benchmarking."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.violations.append(f"Line {node.lineno}: dunder attribute access is blocked.")
        self.generic_visit(node)

    @staticmethod
    def _large_multiplier(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value > 2_000_000

    def _validate_import(self, module_name: str, lineno: int) -> None:
        root = module_name.split(".")[0]
        if root in DENIED_IMPORT_ROOTS:
            self.violations.append(f"Line {lineno}: import `{root}` is denied.")
        elif root and root not in ALLOWED_IMPORT_ROOTS and root not in OPTIONAL_IMPORT_ROOTS:
            self.violations.append(
                f"Line {lineno}: import `{root}` is outside the benchmark allowlist."
            )


def validate_code_for_execution(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError at line {exc.lineno}: {exc.msg}"]
    validator = SafetyValidator()
    validator.visit(tree)
    return validator.violations


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS and root not in OPTIONAL_IMPORT_ROOTS:
        raise ImportError(f"Import `{root}` is not allowed in benchmark mode.")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def build_safe_globals() -> Dict[str, Any]:
    safe_builtins: Dict[str, Any] = {
        name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES if hasattr(builtins, name)
    }
    safe_builtins["__build_class__"] = builtins.__build_class__
    safe_builtins["__import__"] = _safe_import
    return {
        "__builtins__": safe_builtins,
        "__name__": "__user_code__",
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        "Set": Set,
        "Optional": Optional,
    }
