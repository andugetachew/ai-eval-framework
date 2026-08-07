# AI Eval Framework

A pluggable framework for evaluating LLM outputs — combining rule-based
metrics, embedding-based semantic similarity, and LLM-as-judge scoring
behind a single, swappable `Scorer` interface.

Built as a reusable evaluation service: submit a dataset of
input/output/expected-output triples, pick which scorers to run, and get
back per-item scores plus persisted run history.

## Why

Most eval tooling picks one scoring strategy and hard-codes it. This
framework treats scoring strategy as pluggable: exact-match and semantic
similarity give cheap, deterministic signal; an LLM judge gives nuanced,
rubric-based judgment. Both implement the same `BaseScorer` interface, so
a run can mix and match, and new scorers (e.g. RAG-specific metrics like
faithfulness or context relevance) can be added without touching the API
or persistence layer.

## Architecture

- **`app/core/`** — `BaseScorer` interface, shared `EvalItem`/`ScoreResult`
  data models, and `EvalRunner` (the orchestrator that runs scorers
  concurrently over a dataset and persists results)
- **`app/scorers/`** — pluggable scorer implementations:
  - `ExactMatchScorer` — deterministic string match
  - `SemanticSimilarityScorer` — cosine similarity via
    `sentence-transformers` (all-MiniLM-L6-v2)
  - `LLMJudgeScorer` — Claude-based rubric scoring with structured JSON
    output
- **`app/api/`** — FastAPI routes (`POST /eval-runs`, `GET /eval-runs`,
  `GET /eval-runs/{id}`)
- **`app/db/`** — SQLAlchemy async models (`EvalRun`, `EvalResult`) and
  Alembic migrations, backed by PostgreSQL

## Stack

FastAPI · SQLAlchemy (async) · PostgreSQL · Alembic · sentence-transformers
· Anthropic API · Docker Compose · pytest

## Running locally

```bash
docker-compose up
```

API available at `http://localhost:8010`. Health check: `GET /health`.

## Running tests

```bash
docker-compose exec api pytest -v
```

9/9 passing — unit tests for each scorer (LLM judge tested via mocked
Anthropic client) plus full HTTP integration tests against a real
Postgres test database.

## Example usage

```bash
curl -X POST http://localhost:8010/eval-runs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "example run",
    "scorers": ["exact_match", "semantic_similarity"],
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

- RAG-specific scorers (faithfulness, context relevance)
- Multi-model/prompt comparison mode
- Batch dataset upload
- Report export (HTML/markdown summary)