import csv
import io
import json
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from itertools import combinations

from app.core.stats import bootstrap_confidence_interval, welch_t_test
from app.api.schemas import PairwiseComparison

from app.api.auth import require_api_key
from app.core.models import EvalItem
from app.core.runner import EvalRunner
from app.db.deps import get_session
from app.db.models import EvalRun as EvalRunModel
from app.api.schemas import (
    EvalRunRequest,
    EvalRunOut,
    CompareOut,
    VariantSummary,
    PaginatedEvalRuns,
    MAX_ITEMS_PER_RUN,
    MAX_ITEMS_FOR_LLM_SCORERS,
    LLM_SCORER_NAMES,
)
from app.scorers.registry import build_scorer

router = APIRouter(
    prefix="/eval-runs", tags=["eval-runs"], dependencies=[Depends(require_api_key)]
)


def _validate_item_limits(scorer_names: list[str], item_count: int) -> None:
    if item_count > MAX_ITEMS_PER_RUN:
        raise HTTPException(
            status_code=400,
            detail=f"Too many items: {item_count} exceeds max of {MAX_ITEMS_PER_RUN} per run",
        )
    uses_llm_scorer = any(name in LLM_SCORER_NAMES for name in scorer_names)
    if uses_llm_scorer and item_count > MAX_ITEMS_FOR_LLM_SCORERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"LLM-based scorers ({', '.join(LLM_SCORER_NAMES)}) are capped at "
                f"{MAX_ITEMS_FOR_LLM_SCORERS} items per run to control API cost; "
                f"got {item_count} items"
            ),
        )


@router.post("", response_model=EvalRunOut)
async def create_eval_run(
    payload: EvalRunRequest, session: AsyncSession = Depends(get_session)
):
    _validate_item_limits(payload.scorers, len(payload.items))

    scorers = [build_scorer(name) for name in payload.scorers]
    runner = EvalRunner(scorers)
    items = [EvalItem(**item.model_dump()) for item in payload.items]

    run = await runner.run(session, run_name=payload.name, items=items)
    return run


@router.get("", response_model=PaginatedEvalRuns)
async def list_eval_runs(
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    count_stmt = select(func.count()).select_from(EvalRunModel)
    total = (await session.execute(count_stmt)).scalar_one()

    stmt = (
        select(EvalRunModel)
        .options(selectinload(EvalRunModel.results))
        .order_by(EvalRunModel.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    runs = result.scalars().all()

    return PaginatedEvalRuns(total=total, limit=limit, offset=offset, items=runs)


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
        score_values = [s.score for s in scores]
        avg_score = sum(score_values) / len(score_values)
        pass_count = sum(1 for s in scores if s.passed)
        ci_low, ci_high = bootstrap_confidence_interval(score_values)
        summary.append(
            VariantSummary(
                variant=variant,
                scorer_name=scorer_name,
                avg_score=round(avg_score, 4),
                pass_rate=round(pass_count / len(scores), 4),
                count=len(scores),
                ci_low=round(ci_low, 4),
                ci_high=round(ci_high, 4),
            )
        )

    # Pairwise significance testing: compare every pair of variants that
    # share the same scorer (comparing across different scorers wouldn't
    # be meaningful — they're on different scales/semantics)
    scorer_names = {scorer for (_, scorer) in buckets.keys()}
    pairwise = []
    for scorer_name in scorer_names:
        variants_for_scorer = [
            variant for (variant, s) in buckets.keys() if s == scorer_name
        ]
        for variant_a, variant_b in combinations(sorted(variants_for_scorer), 2):
            scores_a = [s.score for s in buckets[(variant_a, scorer_name)]]
            scores_b = [s.score for s in buckets[(variant_b, scorer_name)]]
            test_result = welch_t_test(scores_a, scores_b)
            pairwise.append(
                PairwiseComparison(
                    scorer_name=scorer_name,
                    variant_a=variant_a,
                    variant_b=variant_b,
                    p_value=test_result["p_value"],
                    significant=test_result["significant"],
                    note=test_result["note"],
                )
            )

    return CompareOut(run_id=run.id, run_name=run.name, summary=summary, pairwise=pairwise)

@router.post("/upload", response_model=EvalRunOut)
async def create_eval_run_from_file(
    name: str = Form(...),
    scorers: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    scorer_names = [s.strip() for s in scorers.split(",") if s.strip()]
    built_scorers = [build_scorer(n) for n in scorer_names]

    raw = (await file.read()).decode("utf-8")
    items: list[EvalItem] = []

    if file.filename.endswith(".jsonl"):
        for line in raw.strip().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(_row_to_item(row))
    elif file.filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            items.append(_row_to_item(row))
    else:
        raise HTTPException(status_code=400, detail="File must be .csv or .jsonl")

    if not items:
        raise HTTPException(status_code=400, detail="No rows found in uploaded file")

    _validate_item_limits(scorer_names, len(items))

    runner = EvalRunner(built_scorers)
    run = await runner.run(session, run_name=name, items=items)
    return run


def _row_to_item(row: dict) -> EvalItem:
    required = {"id", "input", "actual_output"}
    missing = required - row.keys()
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Row missing required fields: {missing}"
        )

    context = row.get("context", "")
    context_list = [c.strip() for c in context.split("|") if c.strip()] if context else []

    return EvalItem(
        id=row["id"],
        input=row["input"],
        actual_output=row["actual_output"],
        expected_output=row.get("expected_output") or None,
        context=context_list,
        variant=row.get("variant") or None,
    )

from celery.result import AsyncResult

from app.core.celery_app import celery_app
from app.core.tasks import run_eval_batch
from app.api.schemas import BatchEvalRunRequest, BatchTaskAccepted, BatchTaskStatus


@router.post("/batch", response_model=BatchTaskAccepted, status_code=202)
async def create_eval_run_batch(payload: BatchEvalRunRequest):
    # No item-count cap here — that's the whole point of the async path.
    # LLM-scorer cost guardrail still applies, just checked before queuing.
    llm_scorer_used = any(name in LLM_SCORER_NAMES for name in payload.scorers)
    if llm_scorer_used and len(payload.items) > 500:
        raise HTTPException(
            status_code=400,
            detail="Even async batches are capped at 500 items when using LLM-based scorers",
        )

    items_data = [item.model_dump() for item in payload.items]
    task = run_eval_batch.delay(payload.name, payload.scorers, items_data)
    return BatchTaskAccepted(task_id=task.id)


@router.get("/batch/{task_id}/status", response_model=BatchTaskStatus)
async def get_batch_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return BatchTaskStatus(task_id=task_id, state="PENDING")
    if result.state == "STARTED":
        return BatchTaskStatus(task_id=task_id, state="STARTED")
    if result.state == "SUCCESS":
        return BatchTaskStatus(task_id=task_id, state="SUCCESS", run_id=result.result)
    if result.state == "FAILURE":
        return BatchTaskStatus(task_id=task_id, state="FAILURE", error=str(result.info))

    return BatchTaskStatus(task_id=task_id, state=result.state)