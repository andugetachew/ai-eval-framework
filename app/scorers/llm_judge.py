import json

from anthropic import AsyncAnthropic

from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem, ScoreResult

DEFAULT_RUBRIC = """You are grading an AI model's output.

Input given to the model:
{input}

Model's output:
{actual_output}

{reference_block}

Score the output from 0.0 to 1.0 on overall quality (correctness, relevance, \
clarity). Respond with ONLY a JSON object, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
"""


class LLMJudgeScorer(BaseScorer):
    name = "llm_judge"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        rubric: str = DEFAULT_RUBRIC,
    ):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.rubric = rubric

    async def score(self, item: EvalItem) -> ScoreResult:
        reference_block = (
            f"Reference/expected answer:\n{item.expected_output}"
            if item.expected_output
            else ""
        )
        prompt = self.rubric.format(
            input=item.input,
            actual_output=item.actual_output,
            reference_block=reference_block,
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        try:
            parsed = json.loads(text)
            score = float(parsed["score"])
            reasoning = parsed.get("reasoning")
        except (json.JSONDecodeError, KeyError, ValueError):
            score = 0.0
            reasoning = f"Failed to parse judge output: {text[:200]}"

        return ScoreResult(
            scorer_name=self.name,
            item_id=item.id,
            score=score,
            passed=score >= 0.7,
            reasoning=reasoning,
            raw={"model": self.model},
        )