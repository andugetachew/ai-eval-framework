# AI Eval Framework

A pluggable framework for evaluating LLM outputs — combining rule-based
metrics, n-gram overlap metrics, embedding-based semantic similarity, a
trained classifier, and LLM-as-judge scoring behind a single, swappable
`Scorer` interface. Includes statistical rigor for comparing variants,
synchronous and asynchronous (Celery) execution paths, and batch
dataset ingestion.

**Live demo:** https://ai-eval-framework-35y8.onrender.com (free tier —
may take ~50s to wake up on first request)

## Why

Most eval tooling picks one scoring strategy and hard-codes it. This
framework treats scoring strategy as pluggable: exact-match, BLEU, and
ROUGE give cheap, deterministic signal; semantic similarity and the
trained classifier give a fast learned signal; an LLM judge gives
nuanced, rubric-based judgment. All of them implement the same
`BaseScorer` interface, so a single run can mix and match, and new
scorers can be added without touching the API or persistence layer.

Comparing two prompts or models isn't just "which average is higher" —
`/compare` reports bootstrap confidence intervals per variant and
Welch's t-tests between variant pairs, so you know whether an observed
difference is real or just noise.

## Architecture

- **`app/core/`** — `BaseScorer` interface, shared `EvalItem`/
  `ScoreResult` data models, `EvalRunner` (concurrent scoring +
  persistence orchestrator), `stats.py` (bootstrap CI, Welch's t-test),
  `celery_app.py` / `tasks.py` (async batch execution)
- **`app/scorers/`** — pluggable scorer implementations (see table
  below)
- **`app/ml/`** — training pipeline for the trained classifier scorer:
  synthetic dataset generation, feature engineering, train/test split,
  evaluation report, and the saved model artifact
- **`app/api/`** — FastAPI routes: sync and async eval runs, batch
  upload, pagination, comparison/stats, API-key auth
- **`app/db/`** — SQLAlchemy async models (`EvalRun`, `EvalResult`)
  and Alembic migrations, backed by PostgreSQL

## Scorers

| Scorer | Type | Requires | Cost |
|---|---|---|---|
| `exact_match` | Rule-based | `expected_output` | free |
| `bleu` | N-gram overlap (BLEU) | `expected_output` | free |
| `rouge` | N-gram + LCS overlap (ROUGE-L) | `expected_output` | free |
| `semantic_similarity` | Embedding cosine similarity (sentence-transformers) | `expected_output` | free |
| `trained_classifier` | Trained logistic regression on semantic/lexical features | `expected_output` | free |
| `llm_judge` | LLM-as-judge (Claude), configurable rubric | none | API cost |
| `faithfulness` | LLM-as-judge — checks groundedness in retrieved context | `context` | API cost |
| `context_relevance` | LLM-as-judge — checks retrieved-chunk relevance | `context` | API cost |

`trained_classifier` is trained on three engineered features (semantic
similarity, word overlap, length ratio) over a synthetic labeled
dataset. See `app/ml/generate_training_data.py` and
`app/ml/train_classifier.py` for the full pipeline, including the
train/test split and evaluation report. Retrain with:
`````bash
docker-compose exec api python -m app.ml.generate_training_data
docker-compose exec api python -m app.ml.train_classifier
`````

LLM-based scorers are capped at 25 items per synchronous run (see
Cost controls below) to bound API spend.

## API

| Endpoint | Description |
|---|---|
| `POST /eval-runs` | Run scorers synchronously over a dataset |
| `GET /eval-runs` | List past runs (paginated) |
| `GET /eval-runs/{id}` | Fetch one run and its results |
| `GET /eval-runs/{id}/compare` | Per-variant stats + pairwise significance tests |
| `POST /eval-runs/upload` | Run scorers over an uploaded CSV/JSONL dataset |
| `POST /eval-runs/batch` | Queue a run asynchronously via Celery (202 Accepted) |
| `GET /eval-runs/batch/{task_id}/status` | Poll an async batch job's status |
| `GET /health` | Health check |
| `GET /docs` | Interactive Swagger UI |

### Comparison mode with statistical rigor

Tag items with a `variant` (e.g. which prompt or model produced them)
to compare performance across variants on the same dataset:

`````bash
curl http://localhost:8030/eval-runs/{run_id}/compare
`````

Returns, per variant and scorer: average score, pass rate, and a
bootstrap 95% confidence interval — plus pairwise Welch's t-tests
between every pair of variants sharing a scorer, reporting a p-value
and a plain-language significance verdict.

### Async batch processing

For large datasets, `POST /eval-runs/batch` queues the run via Celery
instead of blocking the request:

`````bash
curl -X POST http://localhost:8030/eval-runs/batch \
  -H "Content-Type: application/json" \
  -d '{"name": "...", "scorers": [...], "items": [...]}'
````//
returns `{"task_id": "...", "status": "queued"}` immediately; poll
`/eval-runs/batch/{task_id}/status` for `state` (`PENDING` /
`STARTED` / `SUCCESS` / `FAILURE`) and the resulting `run_id`.

Requires Redis (Upstash in this setup) and a running Celery worker —
see `docker-compose.yml`. **Not deployed on the live Render demo**:
Render's free tier doesn't support background worker services: this
path is fully working and tested locally via Docker Compose, kept
off the paid tier deliberately.

### Auth

Set `API_KEY` as an environment variable to require an `X-API-Key`
header on all `/eval-runs` routes. Unset (default): auth is disabled,
which is how the local dev setup and current Render deployment run.

### Cost controls

- Max 200 items per synchronous run
- Max 25 items per run when using an LLM-based scorer
  (`llm_judge`, `faithfulness`, `context_relevance`)
- Async batch runs (`/eval-runs/batch`) are capped at 500 items when
  using an LLM-based scorer

### Error visibility

If a scorer fails on a given item (e.g. missing required field), the
failure is captured as a real result with `reasoning` explaining what
went wrong — not silently dropped.

## Stack

FastAPI · SQLAlchemy (async) · PostgreSQL · Alembic · sentence-transformers
· scikit-learn · nltk · rouge-score · scipy · Celery · Redis (Upstash)
· Anthropic API · Docker Compose · pytest

## Running locally

```bash
docker-compose up
```

API available at `http://localhost:8030`. Health check: `GET /health`.
Celery worker and Redis run as additional services in the same
Compose stack for the async batch path.

## Running tests

```bash
docker-compose exec api pytest --cov=app --cov-report=term-missing
```

43/43 passing. One-off scripts (`app/ml/generate_training_data.py`,
`app/ml/train_classifier.py`) and Alembic migrations are excluded from
coverage via `.coveragerc`, since they're run manually rather than
exercised by the application.

## Example usage

```bash
curl -X POST http://localhost:8030/eval-runs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "example run",
    "scorers": ["exact_match", "semantic_similarity", "trained_classifier"],
    "items": [
      {
        "id": "1",
        "input": "What is the capital of France?",
        "actual_output": "Paris is the capital of France.",
        "expected_output": "Paris"
      }
    ]
  }'
```

## Roadmap

- Live testing of LLM-as-judge/RAG scorers (blocked on Anthropic
  billing)
- Report export (HTML/markdown summary)
- Deploy the Celery worker once justified by real batch-size needs
````
