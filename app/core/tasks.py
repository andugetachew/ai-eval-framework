import asyncio

from app.core.celery_app import celery_app
from app.core.models import EvalItem
from app.core.runner import EvalRunner
from app.scorers.registry import build_scorer
from app.db.session import async_session


@celery_app.task(name="run_eval_batch")
def run_eval_batch(run_name: str, scorer_names: list[str], items_data: list[dict]) -> str:
    """
    Celery entrypoint. Celery tasks are sync functions, but our scorers and
    EvalRunner are async — so we spin up a fresh event loop here and run
    the existing async pipeline inside it, unchanged.
    """
    return asyncio.run(_run_eval_batch_async(run_name, scorer_names, items_data))


async def _run_eval_batch_async(
    run_name: str, scorer_names: list[str], items_data: list[dict]
) -> str:
    scorers = [build_scorer(name) for name in scorer_names]
    items = [EvalItem(**data) for data in items_data]
    runner = EvalRunner(scorers)

    async with async_session() as session:
        run = await runner.run(session, run_name=run_name, items=items)
        return run.id