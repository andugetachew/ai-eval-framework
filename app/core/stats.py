import numpy as np
from scipy import stats


def bootstrap_confidence_interval(
    scores: list[float], n_resamples: int = 2000, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Returns a (low, high) confidence interval for the mean of `scores`,
    estimated by resampling with replacement. Works for any score
    distribution shape — doesn't assume normality, unlike a plain
    standard-error calculation.
    """
    if len(scores) < 2:
        # Can't estimate spread from a single point; return it as a
        # degenerate interval rather than fail.
        val = scores[0] if scores else 0.0
        return (val, val)

    arr = np.array(scores)
    result = stats.bootstrap(
        (arr,), np.mean, n_resamples=n_resamples, confidence_level=confidence,
        method="percentile",
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def welch_t_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """
    Welch's t-test: tests whether two variants' mean scores differ
    significantly, without assuming equal variance between them (safer
    default than Student's t-test when comparing two different prompts/
    models, since their score distributions won't generally match).

    Returns p_value and a plain-language significance verdict at p<0.05.
    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {
            "p_value": None,
            "significant": None,
            "note": "Need at least 2 items per variant to test significance",
        }

    t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    return {
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < 0.05),
        "note": (
            "Difference is statistically significant (p < 0.05)"
            if p_value < 0.05
            else "Difference is not statistically significant at p < 0.05 — "
                 "could be noise given the sample size"
        ),
    }