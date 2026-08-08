from pydantic import BaseModel, ConfigDict

MAX_ITEMS_PER_RUN = 200
MAX_ITEMS_FOR_LLM_SCORERS = 25
LLM_SCORER_NAMES = {"llm_judge", "faithfulness", "context_relevance"}


class EvalItemIn(BaseModel):
    id: str
    input: str
    actual_output: str
    expected_output: str | None = None
    context: list[str] = []
    variant: str | None = None


class EvalRunRequest(BaseModel):
    name: str
    scorers: list[str]
    items: list[EvalItemIn]


class ScoreResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    item_id: str
    variant: str | None
    scorer_name: str
    score: float
    passed: bool | None
    reasoning: str | None


class EvalRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    results: list[ScoreResultOut]


class PaginatedEvalRuns(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[EvalRunOut]


class VariantSummary(BaseModel):
    variant: str
    scorer_name: str
    avg_score: float
    pass_rate: float
    count: int


class CompareOut(BaseModel):
    run_id: str
    run_name: str
    summary: list[VariantSummary]

class BatchEvalRunRequest(BaseModel):
    name: str
    scorers: list[str]
    items: list[EvalItemIn]


class BatchTaskAccepted(BaseModel):
    task_id: str
    status: str = "queued"


class BatchTaskStatus(BaseModel):
    task_id: str
    state: str  # PENDING, STARTED, SUCCESS, FAILURE
    run_id: str | None = None
    error: str | None = None