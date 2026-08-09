import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem, ScoreResult

# BLEU/ROUGE tokenization needs the 'punkt' tokenizer data; download once
# at import time (cached after first run, no-op on subsequent starts).
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class BLEUScorer(BaseScorer):
    """
    N-gram precision overlap between actual and expected output — the
    classic machine translation metric. Rewards exact phrasing matches;
    penalizes paraphrasing even when meaning is preserved. Useful as a
    cheap, deterministic baseline alongside semantic_similarity, but
    known to correlate poorly with human judgment for open-ended text.
    """

    name = "bleu"

    async def score(self, item: EvalItem) -> ScoreResult:
        if item.expected_output is None:
            raise ValueError("BLEUScorer requires expected_output")

        reference = [item.expected_output.split()]
        candidate = item.actual_output.split()

        smoothing = SmoothingFunction().method1
        score = sentence_bleu(reference, candidate, smoothing_function=smoothing)

        return ScoreResult(
            scorer_name=self.name,
            item_id=item.id,
            variant=item.variant,
            score=score,
            passed=score >= 0.5,
        )


class ROUGEScorer(BaseScorer):
    """
    Recall-oriented n-gram + longest-common-subsequence overlap
    (ROUGE-L), commonly used for summarization eval. Reports the F1
    component as a single 0.0-1.0 score for consistency with the other
    scorers here.
    """

    name = "rouge"

    def __init__(self):
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    async def score(self, item: EvalItem) -> ScoreResult:
        if item.expected_output is None:
            raise ValueError("ROUGEScorer requires expected_output")

        scores = self._scorer.score(item.expected_output, item.actual_output)
        f1 = scores["rougeL"].fmeasure

        return ScoreResult(
            scorer_name=self.name,
            item_id=item.id,
            variant=item.variant,
            score=f1,
            passed=f1 >= 0.5,
            raw={"precision": scores["rougeL"].precision, "recall": scores["rougeL"].recall},
        )