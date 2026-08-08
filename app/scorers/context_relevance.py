import json

from anthropic import AsyncAnthropic

from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem, ScoreResult

CONTEXT_RELEVANCE_PROMPT = """You are checking whether retrieved context \
chunks are relevant to a user's query.

Query:
{input}

Retrieved context chunks:
{context}

Score from 0.0 to 1.0: 1.0 means all chunks are directly relevant and \
useful for answering the query. 0.0 means the chunks are unrelated noise. \
Respond with ONLY a JSON object:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
"""


class ContextRelevanceScorer(BaseScorer):
    name = "context_relevance"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def score(self, item: EvalItem) -> ScoreResult:
        if not item.context:
            raise ValueError("ContextRelevanceScorer requires item.context (retrieved chunks)")

        prompt = CONTEXT_RELEVANCE_PROMPT.format(
            input=item.input,
            context="\n---\n".join(item.context),
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