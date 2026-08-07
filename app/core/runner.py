import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.base_scorer import BaseScorer
from app.core.models import EvalItem
from app.db.models import EvalRun, EvalResult


class EvalRunner:
    def __init__(self, scorers: list[BaseScorer]):
        self.scorers = scorers

    async def run(
        self, session: AsyncSession, run_name: str, items: list[EvalItem]
    ) -> EvalRun:
        run = EvalRun(name=run_name, status="running")
        session.add(run)
        await session.flush()  # get run.id

        tasks = [
            scorer.score(item) for item in items for scorer in self.scorers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            session.add(
                EvalResult(
                    run_id=run.id,
                    item_id=result.item_id,
                    scorer_name=result.scorer_name,
                    score=result.score,
                    passed=result.passed,
                    reasoning=result.reasoning,
                    raw=result.raw,
                )
            )

        run.status = "completed"
        await session.commit()

        # Re-fetch with results eagerly loaded in one awaited query —
        # avoids ever touching the lazy `.results` attribute directly.
        stmt = (
            select(EvalRun)
            .where(EvalRun.id == run.id)
            .options(selectinload(EvalRun.results))
        )
        result = await session.execute(stmt)
        return result.scalar_one()