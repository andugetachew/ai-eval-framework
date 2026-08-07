from app.core.base_scorer import BaseScorer
from app.core.config import settings
from app.scorers.exact_match import ExactMatchScorer
from app.scorers.semantic_similarity import SemanticSimilarityScorer
from app.scorers.llm_judge import LLMJudgeScorer


def build_scorer(name: str) -> BaseScorer:
    if name == "exact_match":
        return ExactMatchScorer()
    if name == "semantic_similarity":
        return SemanticSimilarityScorer()
    if name == "llm_judge":
        return LLMJudgeScorer(api_key=settings.anthropic_api_key)
    raise ValueError(f"Unknown scorer: {name}")