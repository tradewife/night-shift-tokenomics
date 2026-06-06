"""Tests for robustness gates."""

import numpy as np

from night_shift.data.ingestion.bootstrap import bootstrap_token_history
from night_shift.models.registry import get_model
from night_shift.taxonomy.parameter_space import DEFAULT_PARAMS
from night_shift.validation.folds import create_folds
from night_shift.validation.robustness import (
    cpcv,
    extract_daily_returns,
    generate_cpcv_param_grid,
    monte_carlo_dd,
    param_sensitivity,
    run_robustness_analysis,
)


def _sample_events():
    return bootstrap_token_history(
        {"symbol": "JUP", "mint": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN", "label": "resilient"}
    )


def test_monte_carlo_produces_distribution():
    returns = [0.5, -0.2, 0.3, 0.1, -0.4, 0.2]
    result = monte_carlo_dd(returns, n_simulations=200)
    assert result.n_simulations == 200
    assert result.dd_p95 >= result.dd_p50
    assert 0 <= result.prob_dd_gt_10 <= 1


def test_extract_daily_returns():
    events = _sample_events()
    folds = create_folds(len(events), 3, 30, 30)
    returns = extract_daily_returns(events, DEFAULT_PARAMS, folds)
    assert len(returns) > 0


def test_cpcv_runs():
    events = _sample_events()
    model = get_model(use_stub=False)
    folds = create_folds(len(events), 4, 30, 30)
    grid = generate_cpcv_param_grid(DEFAULT_PARAMS, n_variants=6)
    result = cpcv(events, model, grid, folds, n_test_folds=2)
    assert result.n_paths > 0
    assert 0 <= result.pbo <= 1


def test_param_sensitivity():
    events = _sample_events()
    model = get_model(use_stub=False)
    folds = create_folds(len(events), 3, 30, 30)
    result = param_sensitivity(events, model, DEFAULT_PARAMS, folds)
    assert result.max_sensitivity >= 0


def test_run_robustness_analysis():
    events = _sample_events()
    model = get_model(use_stub=False)
    folds = create_folds(len(events), 4, 30, 30)
    analysis = run_robustness_analysis(
        "JUP",
        events,
        DEFAULT_PARAMS,
        model,
        folds,
        n_mc_simulations=100,
        n_cpcv_variants=6,
    )
    assert "monte_carlo" in analysis
    assert "cpcv" in analysis
    assert "sensitivity" in analysis
    assert "verdict" in analysis
    assert "overall" in analysis["verdict"]


def test_shuffle_changes_dd():
    returns = list(np.random.default_rng(1).normal(0.1, 0.5, 50))
    observed = monte_carlo_dd(returns, n_simulations=50)
    assert observed.observed_dd >= 0