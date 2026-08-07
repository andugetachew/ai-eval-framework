from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem, ScoreResult


class ExactMatchScorer(BaseScorer):
    name = "exact_match"

    async def score(self, item: EvalItem) -> ScoreResult:
        if item.expected_output is None:
            raise ValueError("ExactMatchScorer requires expected_output")

        is_match = item.actual_output.strip() == item.expected_output.strip()
        return ScoreResult(
            scorer_name=self.name,
            item_id=item.id,
            score=1.0 if is_match else 0.0,
            passed=is_match,
        )