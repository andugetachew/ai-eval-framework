import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem, ScoreResult
from app.db.models import EvalRun, EvalResult


async def _safe_score(scorer: BaseScorer, item: EvalItem) -> ScoreResult:
    try:
        return await scorer.score(item)
    except Exception as e:
        return ScoreResult(
            scorer_name=scorer.name,
            item_id=item.id,
            variant=item.variant,
            score=0.0,
            passed=None,
            reasoning=f"Scorer error: {e}",
            raw={"error": True},
        )


class EvalRunner:
    def __init__(self, scorers: list[BaseScorer]):
        self.scorers = scorers

    async def run(
        self, session: AsyncSession, run_name: str, items: list[EvalItem]
    ) -> EvalRun:
        run = EvalRun(name=run_name, status="running")
        session.add(run)
        await session.flush()

        tasks = [
            _safe_score(scorer, item) for item in items for scorer in self.scorers
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            session.add(
                EvalResult(
                    run_id=run.id,
                    item_id=result.item_id,
                    variant=result.variant,
                    scorer_name=result.scorer_name,
                    score=result.score,
                    passed=result.passed,
                    reasoning=result.reasoning,
                    raw=result.raw,
                )
            )

        run.status = "completed"
        await session.commit()

        stmt = (
            select(EvalRun)
            .where(EvalRun.id == run.id)
            .options(selectinload(EvalRun.results))
        )
        result = await session.execute(stmt)
        return result.scalar_one()