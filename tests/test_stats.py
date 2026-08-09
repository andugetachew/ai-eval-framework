from app.core.stats import bootstrap_confidence_interval, welch_t_test


def test_bootstrap_ci_returns_reasonable_bounds():
    scores = [0.9, 0.85, 0.95, 0.88, 0.92]
    low, high = bootstrap_confidence_interval(scores)
    assert low < sum(scores) / len(scores) < high


def test_bootstrap_ci_single_score_returns_degenerate_interval():
    low, high = bootstrap_confidence_interval([0.75])
    assert low == high == 0.75


def test_bootstrap_ci_empty_returns_zero():
    low, high = bootstrap_confidence_interval([])
    assert low == high == 0.0


def test_welch_t_test_detects_significant_difference():
    scores_a = [0.9, 0.85, 0.95, 0.88, 0.92]
    scores_b = [0.6, 0.55, 0.65, 0.58, 0.62]
    result = welch_t_test(scores_a, scores_b)
    assert result["significant"] is True
    assert result["p_value"] < 0.05


def test_welch_t_test_no_difference_not_significant():
    scores_a = [0.8, 0.82, 0.79, 0.81, 0.80]
    scores_b = [0.79, 0.81, 0.80, 0.82, 0.78]
    result = welch_t_test(scores_a, scores_b)
    assert result["significant"] is False


def test_welch_t_test_insufficient_data_returns_none():
    result = welch_t_test([0.9], [0.5])
    assert result["p_value"] is None
    assert result["significant"] is None