import pytest
from unittest.mock import AsyncMock, patch

from app.core.models import EvalItem
from app.scorers.exact_match import ExactMatchScorer
from app.scorers.semantic_similarity import SemanticSimilarityScorer
from app.scorers.llm_judge import LLMJudgeScorer
from app.scorers.faithfulness import FaithfulnessScorer
from app.scorers.context_relevance import ContextRelevanceScorer
from app.scorers.registry import build_scorer
from app.core.base_scorer import BaseScorer



@pytest.mark.asyncio
async def test_exact_match_pass():
    scorer = ExactMatchScorer()
    item = EvalItem(id="1", input="q", actual_output="Paris", expected_output="Paris")
    result = await scorer.score(item)
    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.asyncio
async def test_exact_match_fail():
    scorer = ExactMatchScorer()
    item = EvalItem(id="1", input="q", actual_output="Paris", expected_output="London")
    result = await scorer.score(item)
    assert result.score == 0.0
    assert result.passed is False


@pytest.mark.asyncio
async def test_exact_match_requires_expected_output():
    scorer = ExactMatchScorer()
    item = EvalItem(id="1", input="q", actual_output="Paris", expected_output=None)
    with pytest.raises(ValueError):
        await scorer.score(item)


@pytest.mark.asyncio
async def test_semantic_similarity_high_for_similar_meaning():
    scorer = SemanticSimilarityScorer()
    item = EvalItem(
        id="1",
        input="q",
        actual_output="Paris is the capital of France.",
        expected_output="The capital of France is Paris.",
    )
    result = await scorer.score(item)
    assert result.score > 0.8


@pytest.mark.asyncio
async def test_semantic_similarity_low_for_unrelated():
    scorer = SemanticSimilarityScorer()
    item = EvalItem(
        id="1", input="q", actual_output="Paris is a city.", expected_output="Bananas are yellow."
    )
    result = await scorer.score(item)
    assert result.score < 0.5


@pytest.mark.asyncio
async def test_llm_judge_parses_valid_json_response():
    scorer = LLMJudgeScorer(api_key="fake-key")

    mock_message = AsyncMock()
    mock_message.content = [AsyncMock(text='{"score": 0.9, "reasoning": "Correct and clear"}')]
    mock_create = AsyncMock(return_value=mock_message)

    with patch.object(scorer.client.messages, "create", new=mock_create):
        item = EvalItem(id="1", input="q", actual_output="Paris", expected_output="Paris")
        result = await scorer.score(item)

    assert result.score == 0.9
    assert result.reasoning == "Correct and clear"
    assert result.passed is True


@pytest.mark.asyncio
async def test_llm_judge_handles_malformed_response():
    scorer = LLMJudgeScorer(api_key="fake-key")

    mock_message = AsyncMock()
    mock_message.content = [AsyncMock(text="not valid json")]
    mock_create = AsyncMock(return_value=mock_message)

    with patch.object(scorer.client.messages, "create", new=mock_create):
        item = EvalItem(id="1", input="q", actual_output="Paris", expected_output="Paris")
        result = await scorer.score(item)

    assert result.score == 0.0
    assert "Failed to parse" in result.reasoning


@pytest.mark.asyncio
async def test_faithfulness_requires_context():
    scorer = FaithfulnessScorer(api_key="fake-key")
    item = EvalItem(id="1", input="q", actual_output="Paris", context=[])
    with pytest.raises(ValueError):
        await scorer.score(item)


@pytest.mark.asyncio
async def test_faithfulness_parses_valid_response():
    scorer = FaithfulnessScorer(api_key="fake-key")
    mock_message = AsyncMock()
    mock_message.content = [AsyncMock(text='{"score": 0.85, "reasoning": "Grounded in context"}')]
    mock_create = AsyncMock(return_value=mock_message)

    with patch.object(scorer.client.messages, "create", new=mock_create):
        item = EvalItem(
            id="1", input="q", actual_output="Paris is the capital.",
            context=["Paris is the capital of France."],
        )
        result = await scorer.score(item)

    assert result.score == 0.85
    assert result.passed is True


@pytest.mark.asyncio
async def test_context_relevance_requires_context():
    scorer = ContextRelevanceScorer(api_key="fake-key")
    item = EvalItem(id="1", input="q", actual_output="Paris", context=[])
    with pytest.raises(ValueError):
        await scorer.score(item)


@pytest.mark.asyncio
async def test_context_relevance_parses_valid_response():
    scorer = ContextRelevanceScorer(api_key="fake-key")
    mock_message = AsyncMock()
    mock_message.content = [AsyncMock(text='{"score": 0.6, "reasoning": "Partially relevant"}')]
    mock_create = AsyncMock(return_value=mock_message)

    with patch.object(scorer.client.messages, "create", new=mock_create):
        item = EvalItem(
            id="1", input="capital of France?", actual_output="Paris",
            context=["The Eiffel Tower is in Paris.", "France is in Europe."],
        )
        result = await scorer.score(item)

    assert result.score == 0.6
    assert result.passed is False


def test_registry_builds_all_known_scorers():
    for name in ["exact_match", "semantic_similarity", "llm_judge", "faithfulness", "context_relevance"]:
        scorer = build_scorer(name)
        assert isinstance(scorer, BaseScorer)
        assert scorer.name == name


def test_registry_rejects_unknown_scorer():
    with pytest.raises(ValueError):
        build_scorer("not_a_real_scorer")

from app.scorers.bleu_rouge import BLEUScorer, ROUGEScorer


@pytest.mark.asyncio
async def test_bleu_requires_expected_output():
    scorer = BLEUScorer()
    item = EvalItem(id="1", input="q", actual_output="Paris", expected_output=None)
    with pytest.raises(ValueError):
        await scorer.score(item)


@pytest.mark.asyncio
async def test_bleu_high_score_for_exact_match():
    scorer = BLEUScorer()
    item = EvalItem(
        id="1", input="q", actual_output="the cat sat on the mat",
        expected_output="the cat sat on the mat",
    )
    result = await scorer.score(item)
    assert result.score > 0.9


@pytest.mark.asyncio
async def test_bleu_low_score_for_unrelated_text():
    scorer = BLEUScorer()
    item = EvalItem(
        id="1", input="q", actual_output="completely different words here",
        expected_output="the cat sat on the mat",
    )
    result = await scorer.score(item)
    assert result.score < 0.3


@pytest.mark.asyncio
async def test_rouge_requires_expected_output():
    scorer = ROUGEScorer()
    item = EvalItem(id="1", input="q", actual_output="Paris", expected_output=None)
    with pytest.raises(ValueError):
        await scorer.score(item)


@pytest.mark.asyncio
async def test_rouge_high_score_for_exact_match():
    scorer = ROUGEScorer()
    item = EvalItem(
        id="1", input="q", actual_output="the cat sat on the mat",
        expected_output="the cat sat on the mat",
    )
    result = await scorer.score(item)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_rouge_partial_score_for_paraphrase():
    scorer = ROUGEScorer()
    item = EvalItem(
        id="1", input="q", actual_output="The cat sat on the mat",
        expected_output="A cat was sitting on the mat",
    )
    result = await scorer.score(item)
    assert 0.3 < result.score < 0.9