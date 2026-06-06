"""Tests for the real tokenomics simulator."""

from night_shift.data.ingestion.bootstrap import bootstrap_token_history
from night_shift.simulation.tokenomics_sim import evaluate_design
from night_shift.taxonomy.parameter_space import DEFAULT_PARAMS


def _holder_heavy_params():
    p = dict(DEFAULT_PARAMS)
    p.update({"holder_pct": 40, "dev_pct": 5, "burn_pct": 15, "treasury_pct": 40})
    return p


def _extractive_params():
    p = dict(DEFAULT_PARAMS)
    p.update({"holder_pct": 5, "dev_pct": 40, "burn_pct": 0, "treasury_pct": 55})
    return p


def test_holder_heavy_beats_extractive():
    events = bootstrap_token_history(
        {"symbol": "JUP", "mint": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN", "label": "resilient"}
    )
    good = evaluate_design(events, _holder_heavy_params(), 30, 120)
    bad = evaluate_design(events, _extractive_params(), 30, 120)
    assert good.score > bad.score
    assert good.pnl > bad.pnl


def test_insufficient_data_returns_zero():
    events = bootstrap_token_history(
        {"symbol": "X", "mint": "X" * 44, "label": "unknown"}, days=10
    )
    result = evaluate_design(events, DEFAULT_PARAMS, 0, 2)
    assert result.event_count == 0
    assert result.outcomes.get("insufficient_data") == 1