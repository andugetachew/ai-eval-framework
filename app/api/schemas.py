from pydantic import BaseModel, ConfigDict


class EvalItemIn(BaseModel):
    id: str
    input: str
    actual_output: str
    expected_output: str | None = None


class EvalRunRequest(BaseModel):
    name: str
    scorers: list[str]
    items: list[EvalItemIn]


class ScoreResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    item_id: str
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