"""Static analysis package for Complexity Lab."""

from analyzer.ast_analyzer import analyze_code
from analyzer.models import (
    AlgorithmPattern,
    AntiPattern,
    FunctionAnalysis,
    LineFinding,
    OptimizationTarget,
    StaticAnalysisResult,
)

__all__ = [
    "AlgorithmPattern",
    "AntiPattern",
    "FunctionAnalysis",
    "LineFinding",
    "OptimizationTarget",
    "StaticAnalysisResult",
    "analyze_code",
]
