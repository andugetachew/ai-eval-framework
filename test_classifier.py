import asyncio
from app.scorers.trained_classifier import TrainedClassifierScorer
from app.core.models import EvalItem

async def main():
    scorer = TrainedClassifierScorer()

    good = EvalItem(id="1", input="q", actual_output="Paris is the capital of France.", expected_output="The capital of France is Paris.")
    bad = EvalItem(id="2", input="q", actual_output="Bananas are yellow.", expected_output="The capital of France is Paris.")

    r1 = await scorer.score(good)
    r2 = await scorer.score(bad)
    print("Good pair:", r1.score, r1.passed, r1.raw)
    print("Bad pair:", r2.score, r2.passed, r2.raw)

asyncio.run(main())