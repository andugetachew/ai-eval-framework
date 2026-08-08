import json

from anthropic import AsyncAnthropic

from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem, ScoreResult

FAITHFULNESS_PROMPT = """You are checking whether an AI-generated answer is \
faithful to the retrieved context it was given — i.e. it doesn't state \
anything that isn't supported by the context.

Retrieved context:
{context}

Generated answer:
{actual_output}

Score from 0.0 to 1.0: 1.0 means every claim in the answer is directly \
supported by the context. 0.0 means the answer contradicts or invents \
information not present in the context. Respond with ONLY a JSON object:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
"""


class FaithfulnessScorer(BaseScorer):
    name = "faithfulness"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def score(self, item: EvalItem) -> ScoreResult:
        if not item.context:
            raise ValueError("FaithfulnessScorer requires item.context (retrieved chunks)")

        prompt = FAITHFULNESS_PROMPT.format(
            context="\n---\n".join(item.context),
            actual_output=item.actual_output,
        )
        response = await self.client.messages.create(
            model=self.model, max_tokens=300,
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
            scorer_name=self.name, item_id=item.id, variant=item.variant,
            score=score, passed=score >= 0.7, reasoning=reasoning,
            raw={"model": self.model},
        )