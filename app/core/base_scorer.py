from abc import ABC, abstractmethod

from app.core.models import EvalItem, ScoreResult


class BaseScorer(ABC):
    """
    Every scorer — rule-based or LLM-as-judge — implements this interface.
    That's what makes scorers swappable/pluggable in an eval run.
    """

    name: str

    @abstractmethod
    async def score(self, item: EvalItem) -> ScoreResult:
        ...