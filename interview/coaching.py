"""Local interview simulation and answer grading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from analyzer.models import StaticAnalysisResult
from optimization.planner import OptimizationPlan


@dataclass
class InterviewGrade:
    total_score: int
    rubric: Dict[str, int] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    model_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_follow_up_questions(analysis: StaticAnalysisResult, plan: OptimizationPlan) -> List[str]:
    questions = [
        "Can this be improved further, and what would you change first?",
        "What is the space complexity, including auxiliary data structures and recursion stack?",
        "What edge cases would you test before submitting this solution?",
    ]
    if analysis.metrics.get("max_loop_depth", 0) >= 2:
        questions.append("The nested loop is the main concern. Can you replace the inner scan with a better lookup?")
    if analysis.metrics.get("hash_membership_tests", 0) or analysis.metrics.get("dict_builds", 0):
        questions.append("Why did you choose a set or dictionary, and what memory trade-off does it introduce?")
    if analysis.metrics.get("recursive_calls", 0):
        questions.append("How deep can the recursion go, and do overlapping subproblems require memoization?")
    if analysis.metrics.get("sort_calls", 0):
        questions.append("Why is sorting acceptable here, and does it change the original output requirements?")
    if plan.optimized_code:
        questions.append("How would you prove the optimized version returns the same result as the original?")
    return questions[:7]


def grade_interview_answer(answer: str, analysis: StaticAnalysisResult, plan: OptimizationPlan) -> InterviewGrade:
    text = (answer or "").lower()
    rubric = {
        "correctness": 0,
        "complexity": 0,
        "edge_cases": 0,
        "tradeoffs": 0,
        "communication": 0,
    }
    strengths: List[str] = []
    improvements: List[str] = []

    if any(token in text for token in ("correct", "invariant", "because", "returns", "case")):
        rubric["correctness"] = 16
        strengths.append("You attempted to justify correctness.")
    else:
        improvements.append("Add a correctness argument or invariant, not only runtime claims.")

    if any(token in text for token in ("o(", "time", "space", "complexity", "linear", "quadratic", "log")):
        rubric["complexity"] = 22
        strengths.append("You discussed complexity explicitly.")
    else:
        improvements.append("State time and space complexity directly.")

    if any(token in text for token in ("empty", "duplicate", "negative", "single", "none", "edge")):
        rubric["edge_cases"] = 18
        strengths.append("You mentioned edge cases.")
    else:
        improvements.append("Mention edge cases such as empty input, duplicates, and boundary sizes.")

    if any(token in text for token in ("trade", "memory", "readability", "set", "dict", "hash", "sort")):
        rubric["tradeoffs"] = 20
        strengths.append("You included trade-off language.")
    else:
        improvements.append("Explain the runtime vs memory/readability trade-off.")

    words = [word for word in text.split() if word.strip()]
    if 35 <= len(words) <= 180:
        rubric["communication"] = 20
        strengths.append("The answer length is interview-friendly.")
    elif words:
        rubric["communication"] = 12
        improvements.append("Make the answer concise enough to say aloud in under one minute.")
    else:
        improvements.append("Write an answer first, then grade it.")

    current = plan.interview_feedback.get("current_explanation", "")
    optimized = plan.interview_feedback.get("optimized_explanation", "")
    model_answer = (
        f"{current} {optimized} I would validate correctness with edge cases and use benchmarks only as supporting "
        "evidence, while relying on asymptotic analysis for the main interview explanation."
    )

    return InterviewGrade(
        total_score=sum(rubric.values()),
        rubric=rubric,
        strengths=strengths,
        improvements=improvements,
        model_answer=model_answer,
    )
