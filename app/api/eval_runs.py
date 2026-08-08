from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.models import EvalItem
from app.core.runner import EvalRunner
from app.db.deps import get_session
from app.db.models import EvalRun as EvalRunModel
from app.api.schemas import EvalRunRequest, EvalRunOut, CompareOut, VariantSummary
from app.scorers.registry import build_scorer

router = APIRouter(prefix="/eval-runs", tags=["eval-runs"])


@router.post("", response_model=EvalRunOut)
async def create_eval_run(
    payload: EvalRunRequest, session: AsyncSession = Depends(get_session)
):
    scorers = [build_scorer(name) for name in payload.scorers]
    runner = EvalRunner(scorers)
    items = [EvalItem(**item.model_dump()) for item in payload.items]

    run = await runner.run(session, run_name=payload.name, items=items)
    return run


@router.get("", response_model=list[EvalRunOut])
async def list_eval_runs(session: AsyncSession = Depends(get_session)):
    stmt = select(EvalRunModel).options(selectinload(EvalRunModel.results))
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/{run_id}", response_model=EvalRunOut)
async def get_eval_run(run_id: str, session: AsyncSession = Depends(get_session)):
    stmt = (
        select(EvalRunModel)
        .where(EvalRunModel.id == run_id)
        .options(selectinload(EvalRunModel.results))
    )
    result = await session.execute(stmt)
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return run


@router.get("/{run_id}/compare", response_model=CompareOut)
async def compare_variants(run_id: str, session: AsyncSession = Depends(get_session)):
    stmt = (
        select(EvalRunModel)
        .where(EvalRunModel.id == run_id)
        .options(selectinload(EvalRunModel.results))
    )
    result = await session.execute(stmt)
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")

    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for r in run.results:
        variant = r.variant or "default"
        buckets[(variant, r.scorer_name)].append(r)

    summary = []
    for (variant, scorer_name), scores in buckets.items():
        avg_score = sum(s.score for s in scores) / len(scores)
        pass_count = sum(1 for s in scores if s.passed)
        summary.append(
            VariantSummary(
                variant=variant,
                scorer_name=scorer_name,
                avg_score=round(avg_score, 4),
                pass_rate=round(pass_count / len(scores), 4),
                count=len(scores),
            )
        )

    return CompareOut(run_id=run.id, run_name=run.name, summary=summary)