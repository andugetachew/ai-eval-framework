import os

import joblib
from sentence_transformers import SentenceTransformer, util

from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem, ScoreResult
from app.ml.features import word_overlap_ratio, length_ratio

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "classifier.joblib")

_embedding_model = None
_classifier = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = joblib.load(_MODEL_PATH)
    return _classifier


class TrainedClassifierScorer(BaseScorer):
    """
    A logistic regression classifier trained on (semantic_similarity,
    word_overlap, length_ratio) features to predict good/bad output
    matches. Unlike the other scorers here, this one is an actual
    trained model rather than a pretrained-model call or a fixed
    formula — see app/ml/train_classifier.py for the training pipeline.
    Deterministic and free to run (no API cost), making it a fast
    stand-in for llm_judge on large batches.
    """

    name = "trained_classifier"

    async def score(self, item: EvalItem) -> ScoreResult:
        if item.expected_output is None:
            raise ValueError("TrainedClassifierScorer requires expected_output")

        model = _get_embedding_model()
        clf = _get_classifier()

        emb = model.encode([item.actual_output, item.expected_output], convert_to_tensor=True)
        semantic_sim = util.cos_sim(emb[0], emb[1]).item()
        word_overlap = word_overlap_ratio(item.actual_output, item.expected_output)
        length_ratio_val = length_ratio(item.actual_output, item.expected_output)

        features = [[semantic_sim, word_overlap, length_ratio_val]]
        prediction = clf.predict(features)[0]
        confidence = clf.predict_proba(features)[0][1]

        return ScoreResult(
            scorer_name=self.name,
            item_id=item.id,
            variant=item.variant,
            score=float(confidence),
            passed=bool(prediction == 1),
            raw={
                "semantic_similarity": semantic_sim,
                "word_overlap": word_overlap,
                "length_ratio": length_ratio_val,
            },
        )