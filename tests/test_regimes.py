"""Tests for cross-regime consistency gate (SPEC §4 Gate 2)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from night_shift.models.registry import get_model
from night_shift.taxonomy.parameter_space import DEFAULT_PARAMS
from night_shift.validation.evaluate import evaluate_candidate
from night_shift.validation.folds import create_folds
from night_shift.validation.gates import RegimeGate
from night_shift.validation.regimes import (
    BEAR_HOLDER_GROWTH_MAX,
    BEAR_VOLUME_TREND_MAX,
    BULL_HOLDER_GROWTH_MIN,
    BULL_VOLUME_TREND_MIN,
    HIGH_VOL_CV_THRESHOLD,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_HIGH_VOL,
    REGIME_SIDEWAYS,
    REGIME_UNKNOWN,
    annotate_fold_regimes,
    build_regime_summary,
    check_regime_gate,
    classify_regime,
    extract_regime_features,
    tag_window_regime,
)


def _daily_row(day: int, volume: float, holders: int, top10: float = 40.0) -> dict:
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "timestamp": ts,
        "day": day,
        "holder_count": holders,
        "top10_holder_pct": top10,
        "transfer_volume": volume,
        "mint_amount": 0.0,
        "burn_amount": 0.0,
        "treasury_inflow": volume * 0.01,
        "active_addresses": max(10, int(holders * 0.1)),
    }


def _segment_bull(start: int, length: int) -> list[dict]:
    rows = []
    holders = 1000
    for i in range(length):
        day = start + i
        volume = 10_000 * float(np.exp(0.09 * i))
        holders += 12
        rows.append(_daily_row(day, volume, holders, 35.0))
    return rows


def _segment_bear(start: int, length: int) -> list[dict]:
    rows = []
    holders = 2500
    for i in range(length):
        day = start + i
        volume = 250_000 * float(np.exp(-0.09 * i))
        holders = max(200, holders - 15)
        rows.append(_daily_row(day, volume, holders, 55.0))
    return rows


def _segment_sideways(start: int, length: int) -> list[dict]:
    rows = []
    holders = 1500
    for i in range(length):
        day = start + i
        volume = 60_000.0
        rows.append(_daily_row(day, volume, holders, 42.0))
    return rows


def _segment_high_vol(start: int, length: int) -> list[dict]:
    rows = []
    holders = 1200
    for i in range(length):
        day = start + i
        volume = 8_000.0 if i % 2 == 0 else 220_000.0
        holders += 1 if i % 3 == 0 else -1
        rows.append(_daily_row(day, volume, max(100, holders), 48.0))
    return rows


def regime_fixture_dataframe(days: int = 180) -> pd.DataFrame:
    """Hand-crafted history with four auditable regime segments (post-warmup)."""
    warmup = 30
    seg = (days - warmup) // 4
    rows = [_daily_row(i, 40_000, 900 + i, 40.0) for i in range(warmup)]
    rows.extend(_segment_bull(warmup, seg))
    rows.extend(_segment_bear(warmup + seg, seg))
    rows.extend(_segment_sideways(warmup + 2 * seg, seg))
    rows.extend(_segment_high_vol(warmup + 3 * seg, days - warmup - 3 * seg))
    return pd.DataFrame(rows)


def test_classify_regime_deterministic():
    bull_features = {
        "volume_trend": BULL_VOLUME_TREND_MIN + 0.001,
        "holder_growth": BULL_HOLDER_GROWTH_MIN + 0.01,
        "concentration_delta": -1.0,
        "tx_variance": 0.2,
    }
    for _ in range(100):
        assert classify_regime(bull_features) == REGIME_BULL


def test_high_vol_overrides_bull():
    features = {
        "volume_trend": BULL_VOLUME_TREND_MIN + 0.01,
        "holder_growth": BULL_HOLDER_GROWTH_MIN + 0.05,
        "concentration_delta": 0.0,
        "tx_variance": HIGH_VOL_CV_THRESHOLD + 0.05,
    }
    assert classify_regime(features) == REGIME_HIGH_VOL


def test_bear_boundary():
    features = {
        "volume_trend": BEAR_VOLUME_TREND_MAX,
        "holder_growth": BEAR_HOLDER_GROWTH_MAX,
        "concentration_delta": 2.0,
        "tx_variance": 0.3,
    }
    assert classify_regime(features) == REGIME_BEAR


def test_sideways_residual():
    features = {
        "volume_trend": 0.0,
        "holder_growth": 0.0,
        "concentration_delta": 0.0,
        "tx_variance": 0.25,
    }
    assert classify_regime(features) == REGIME_SIDEWAYS


def test_short_window_is_unknown():
    df = regime_fixture_dataframe()
    regime, features = tag_window_regime(df, 0, 5)
    assert regime == REGIME_UNKNOWN
    assert features["window_bars"] == 5.0


def test_fixture_segments_tag_high_vol_and_sideways():
    """Realistic windows reach high_vol/sideways under frozen thresholds; bull/bear via features."""
    df = regime_fixture_dataframe()
    warmup = 30
    seg = (len(df) - warmup) // 4

    side_regime, _ = tag_window_regime(df, warmup + 2 * seg, warmup + 3 * seg)
    hv_regime, _ = tag_window_regime(df, warmup + 3 * seg, len(df))

    assert side_regime == REGIME_SIDEWAYS
    assert hv_regime == REGIME_HIGH_VOL


def test_gate_passes_with_three_profitable_regimes():
    gate = RegimeGate(ENABLED=True, MIN_PROFITABLE_REGIMES=3, MIN_REGIME_SCORE=0.0)
    fold_details = [
        {"regime": REGIME_BULL, "oos_score": 0.5},
        {"regime": REGIME_BEAR, "oos_score": -0.2},
        {"regime": REGIME_BEAR, "oos_score": -0.1},
        {"regime": REGIME_SIDEWAYS, "oos_score": 0.3},
        {"regime": REGIME_HIGH_VOL, "oos_score": 0.2},
    ]
    passed, summary = check_regime_gate(fold_details, gate)
    assert passed
    assert summary["gate_passed"]
    assert summary["profitable_count"] == 3
    assert set(summary["profitable_regimes"]) == {REGIME_BULL, REGIME_SIDEWAYS, REGIME_HIGH_VOL}


def test_gate_fails_with_only_two_profitable_regimes():
    gate = RegimeGate(ENABLED=True, MIN_PROFITABLE_REGIMES=3, MIN_REGIME_SCORE=0.0)
    fold_details = [
        {"regime": REGIME_BULL, "oos_score": 0.5},
        {"regime": REGIME_BEAR, "oos_score": -0.2},
        {"regime": REGIME_SIDEWAYS, "oos_score": 0.3},
        {"regime": REGIME_HIGH_VOL, "oos_score": -0.1},
    ]
    passed, summary = check_regime_gate(fold_details, gate)
    assert not passed
    assert not summary["gate_passed"]
    assert summary["profitable_count"] == 2
    assert summary["failures"]


def test_gate_disabled_always_passes_with_summary():
    gate = RegimeGate(ENABLED=False)
    fold_details = [{"regime": REGIME_BULL, "oos_score": -1.0}]
    passed, summary = check_regime_gate(fold_details, gate)
    assert passed
    assert summary["skipped"]
    assert summary["gate_passed"]


def test_build_regime_summary_by_regime_metrics():
    gate = RegimeGate()
    fold_details = [
        {"regime": REGIME_BULL, "oos_score": 0.4},
        {"regime": REGIME_BULL, "oos_score": 0.1},
        {"regime": REGIME_UNKNOWN, "oos_score": 0.9},
    ]
    summary = build_regime_summary(fold_details, gate)
    bull = summary["by_regime"][REGIME_BULL]
    assert bull["folds"] == 2
    assert bull["best_oos"] == 0.4
    assert bull["profitable"] is True
    assert REGIME_UNKNOWN not in summary["by_regime"]


def test_evaluate_candidate_always_populates_regime_summary():
    df = regime_fixture_dataframe()
    folds = create_folds(len(df), num_folds=5, test_fold_days=30, warmup_bars=30)
    model = get_model(use_stub=True)
    of_config = {
        "max_is_oos_gap": 0.5,
        "min_oos_consistency": 0.0,
        "regime_gate": {"enabled": True, "min_profitable_regimes": 3},
    }

    result = evaluate_candidate(
        df, folds, DEFAULT_PARAMS, "TEST", model, of_config, compute_fragility=False
    )

    assert result.regime_summary
    assert "gate_passed" in result.regime_summary
    assert "by_regime" in result.regime_summary
    assert all("regime" in fold for fold in result.folds)
    assert all("regime_features" in fold for fold in result.folds)


def test_evaluate_candidate_regime_gate_disabled_summary():
    df = regime_fixture_dataframe()
    folds = create_folds(len(df), num_folds=5, test_fold_days=30, warmup_bars=30)
    model = get_model(use_stub=True)
    of_config = {
        "max_is_oos_gap": 0.5,
        "min_oos_consistency": 0.0,
        "regime_gate": {"enabled": False},
    }

    result = evaluate_candidate(
        df, folds, DEFAULT_PARAMS, "TEST", model, of_config, compute_fragility=False
    )

    assert result.regime_summary["skipped"] is True
    assert result.regime_summary["gate_passed"] is True


def test_rejection_reason_appends_regime_failure():
    df = regime_fixture_dataframe()
    folds = create_folds(len(df), num_folds=5, test_fold_days=30, warmup_bars=30)
    model = get_model(use_stub=True)
    of_config = {
        "max_is_oos_gap": -0.01,
        "min_oos_consistency": 0.0,
        "regime_gate": {"enabled": True, "min_profitable_regimes": 3},
    }

    result = evaluate_candidate(
        df, folds, DEFAULT_PARAMS, "TEST", model, of_config, compute_fragility=False
    )

    assert result.rejected
    assert "overfitting_score" in result.rejection_reason
    assert "regime_gate" in result.rejection_reason


def test_annotate_fold_regimes_matches_folds():
    df = regime_fixture_dataframe()
    folds = create_folds(len(df), num_folds=4, test_fold_days=30, warmup_bars=30)
    fold_details = [{"fold": f.fold_num, "oos_score": 0.1} for f in folds]
    annotated = annotate_fold_regimes(df, fold_details, folds)
    assert len(annotated) == len(folds)
    regimes = {a["regime"] for a in annotated}
    assert REGIME_UNKNOWN not in regimes or len(folds) == 0


def test_extract_regime_features_stable():
    df = regime_fixture_dataframe()
    window = df.iloc[30:60]
    a = extract_regime_features(window)
    b = extract_regime_features(window)
    assert a == b