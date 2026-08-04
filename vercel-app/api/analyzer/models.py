"""Typed models shared by the static analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LineFinding:
    line: int
    title: str
    severity: str
    category: str
    detail: str
    suggestion: str


@dataclass
class AlgorithmPattern:
    name: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    interview_note: str = ""


@dataclass
class AntiPattern:
    name: str
    category: str
    severity: str
    message: str
    evidence: str
    suggestion: str
    weight: int


@dataclass
class OptimizationTarget:
    title: str
    impact: str
    effort: str
    rationale: str


@dataclass
class FunctionAnalysis:
    name: str
    lineno: int
    estimated_time: str
    estimated_space: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    anti_patterns: List[AntiPattern] = field(default_factory=list)
    line_findings: List[LineFinding] = field(default_factory=list)
    algorithm_patterns: List[AlgorithmPattern] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticAnalysisResult:
    valid: bool
    raw_code: str
    estimated_time: str = "Unknown"
    estimated_space: str = "Unknown"
    confidence: float = 0.0
    parse_error: Optional[str] = None
    functions: List[FunctionAnalysis] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    anti_patterns: List[AntiPattern] = field(default_factory=list)
    optimization_targets: List[OptimizationTarget] = field(default_factory=list)
    line_findings: List[LineFinding] = field(default_factory=list)
    algorithm_patterns: List[AlgorithmPattern] = field(default_factory=list)
    call_graph: Dict[str, List[str]] = field(default_factory=dict)
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_bottlenecks(self) -> bool:
        return bool(self.anti_patterns)
