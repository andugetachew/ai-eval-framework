import pytest
from unittest.mock import AsyncMock, patch

from app.core.models import EvalItem
from app.scorers.exact_match import ExactMatchScorer
from app.scorers.semantic_similarity import SemanticSimilarityScorer
from app.scorers.llm_judge import LLMJudgeScorer


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