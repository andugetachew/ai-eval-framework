from sentence_transformers import SentenceTransformer, util

from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem, ScoreResult

_model = None


def _get_model() -> SentenceTransformer:
    # Loaded once and reused — this is the same model your Knowledge Base
    # API uses, so it's already familiar territory.
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class SemanticSimilarityScorer(BaseScorer):
    name = "semantic_similarity"

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    async def score(self, item: EvalItem) -> ScoreResult:
        if item.expected_output is None:
            raise ValueError("SemanticSimilarityScorer requires expected_output")

        model = _get_model()
        embeddings = model.encode(
            [item.actual_output, item.expected_output], convert_to_tensor=True
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return ScoreResult(
                    scorer_name=self.name,
                    item_id=item.id,
                    variant=item.variant,
                    score=similarity,
                    passed=similarity >= self.threshold,
                    raw={"threshold": self.threshold},
                )