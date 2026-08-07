from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import EvalItem
from app.core.runner import EvalRunner
from app.db.deps import get_session
from app.api.schemas import EvalRunRequest, EvalRunOut
from app.scorers.registry import build_scorer
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models import EvalRun as EvalRunModel



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