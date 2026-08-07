from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvalItem:
    """One thing to evaluate: an input, what the model produced, and (optionally) a reference answer."""
    id: str
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """The result of running one scorer against one EvalItem."""
    scorer_name: str
    item_id: str
    score: float  # normalized 0.0–1.0
    passed: Optional[bool] = None
    reasoning: Optional[str] = None  # populated by LLM-judge scorers
    raw: dict[str, Any] = field(default_factory=dict)