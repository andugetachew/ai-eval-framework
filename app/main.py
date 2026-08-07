from fastapi import FastAPI

from app.api.eval_runs import router as eval_runs_router

app = FastAPI(title="AI Eval Framework")

app.include_router(eval_runs_router)


@app.get("/health")
async def health():
    return {"status": "ok"}