"""Deterministic market regime tagging and cross-regime consistency gate (SPEC §4 Gate 2)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from night_shift.validation.gates import RegimeGate

# Frozen taxonomy thresholds — do not tune against seed data.
HIGH_VOL_CV_THRESHOLD = 0.55
BULL_VOLUME_TREND_MIN = 0.002
BULL_HOLDER_GROWTH_MIN = 0.02
BEAR_VOLUME_TREND_MAX = -0.002
BEAR_HOLDER_GROWTH_MAX = -0.02
MIN_WINDOW_BARS = 7

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"
REGIME_HIGH_VOL = "high_vol"
REGIME_UNKNOWN = "unknown"

ALL_REGIMES = (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_HIGH_VOL)


def _empty_regime_summary(gate: RegimeGate, *, skipped: bool = False, reason: str = "") -> Dict[str, Any]:
    return {
        "enabled": gate.ENABLED,
        "skipped": skipped,
        "skip_reason": reason,
        "gate_passed": True if skipped or not gate.ENABLED else False,
        "min_profitable_regimes": gate.MIN_PROFITABLE_REGIMES,
        "min_regime_score": gate.MIN_REGIME_SCORE,
        "profitable_regimes": [],
        "profitable_count": 0,
        "by_regime": {},
        "failures": [] if skipped or not gate.ENABLED else ["regime_gate: no fold data"],
    }


def _log1p_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    y = np.log1p(np.maximum(values, 0.0))
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / max(len(values), 1)


def extract_regime_features(window: pd.DataFrame) -> Dict[str, float]:
    """Compute auditable regime features for one test window."""
    volume = window["transfer_volume"].astype(float).to_numpy()
    holders = window["holder_count"].astype(float).to_numpy()
    top10 = window["top10_holder_pct"].astype(float).to_numpy()

    holder_start = holders[0]
    holder_end = holders[-1]
    holder_growth = (holder_end - holder_start) / max(holder_start, 1.0)
    vol_mean = float(np.mean(volume))
    tx_variance = float(np.std(volume) / max(vol_mean, 1e-9))

    return {
        "volume_trend": round(_log1p_slope(volume), 6),
        "holder_growth": round(holder_growth, 6),
        "concentration_delta": round(float(top10[-1] - top10[0]), 6),
        "tx_variance": round(tx_variance, 6),
    }


def classify_regime(features: Dict[str, float]) -> str:
    """
    Priority-ordered regime taxonomy (frozen rules).

    1. high_vol — elevated volume coefficient of variation
    2. bull — rising volume + holder growth
    3. bear — falling volume + holder decline
    4. sideways — residual
    """
    if features["tx_variance"] >= HIGH_VOL_CV_THRESHOLD:
        return REGIME_HIGH_VOL
    if (
        features["volume_trend"] >= BULL_VOLUME_TREND_MIN
        and features["holder_growth"] >= BULL_HOLDER_GROWTH_MIN
    ):
        return REGIME_BULL
    if (
        features["volume_trend"] <= BEAR_VOLUME_TREND_MAX
        and features["holder_growth"] <= BEAR_HOLDER_GROWTH_MAX
    ):
        return REGIME_BEAR
    return REGIME_SIDEWAYS


def tag_window_regime(events: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[str, Dict[str, float]]:
    """Tag one test window. Returns (regime, features). Short windows -> unknown."""
    window = events.iloc[start_idx:end_idx]
    if len(window) < MIN_WINDOW_BARS:
        return REGIME_UNKNOWN, {
            "volume_trend": 0.0,
            "holder_growth": 0.0,
            "concentration_delta": 0.0,
            "tx_variance": 0.0,
            "window_bars": float(len(window)),
        }

    features = extract_regime_features(window)
    features["window_bars"] = float(len(window))
    return classify_regime(features), features


def build_regime_summary(
    fold_details: List[Dict[str, Any]],
    gate: RegimeGate,
) -> Dict[str, Any]:
    """Aggregate per-fold regime labels into an auditable summary."""
    if not gate.ENABLED:
        return _empty_regime_summary(gate, skipped=True, reason="regime_gate disabled")

    by_regime: Dict[str, Dict[str, Any]] = {}
    for fold in fold_details:
        regime = fold.get("regime", REGIME_UNKNOWN)
        if regime == REGIME_UNKNOWN:
            continue

        oos_score = float(fold.get("oos_score", 0.0))
        entry = by_regime.setdefault(
            regime,
            {"folds": 0, "oos_scores": [], "best_oos": float("-inf"), "profitable": False},
        )
        entry["folds"] += 1
        entry["oos_scores"].append(oos_score)
        entry["best_oos"] = max(entry["best_oos"], oos_score)

    profitable_regimes: List[str] = []
    for regime, entry in sorted(by_regime.items()):
        best_oos = entry["best_oos"] if entry["oos_scores"] else float("-inf")
        entry["best_oos"] = round(best_oos, 4) if best_oos != float("-inf") else None
        entry["profitable"] = best_oos > gate.MIN_REGIME_SCORE
        entry["mean_oos"] = round(float(np.mean(entry["oos_scores"])), 4) if entry["oos_scores"] else None
        del entry["oos_scores"]
        if entry["profitable"]:
            profitable_regimes.append(regime)

    profitable_count = len(profitable_regimes)
    failures: List[str] = []
    if profitable_count < gate.MIN_PROFITABLE_REGIMES:
        failures.append(
            f"regime_gate: profitable_regimes={profitable_count}/{gate.MIN_PROFITABLE_REGIMES} "
            f"[{', '.join(profitable_regimes) or 'none'}] "
            f"breakdown={by_regime}"
        )

    return {
        "enabled": True,
        "skipped": False,
        "skip_reason": "",
        "gate_passed": profitable_count >= gate.MIN_PROFITABLE_REGIMES,
        "min_profitable_regimes": gate.MIN_PROFITABLE_REGIMES,
        "min_regime_score": gate.MIN_REGIME_SCORE,
        "profitable_regimes": profitable_regimes,
        "profitable_count": profitable_count,
        "by_regime": by_regime,
        "failures": failures,
    }


def check_regime_gate(
    fold_details: List[Dict[str, Any]],
    gate: RegimeGate,
) -> Tuple[bool, Dict[str, Any]]:
    """Return (passed, regime_summary). Fails closed when enabled and under threshold."""
    summary = build_regime_summary(fold_details, gate)
    if not gate.ENABLED or summary["skipped"]:
        return True, summary
    return summary["gate_passed"], summary


def annotate_fold_regimes(
    events: pd.DataFrame,
    fold_details: List[Dict[str, Any]],
    folds: List[Any],
) -> List[Dict[str, Any]]:
    """Attach regime labels and features to each fold detail entry."""
    annotated: List[Dict[str, Any]] = []
    for i, fold_detail in enumerate(fold_details):
        detail = dict(fold_detail)
        if i < len(folds):
            regime, features = tag_window_regime(
                events,
                folds[i].test_start_idx,
                folds[i].test_end_idx,
            )
            detail["regime"] = regime
            detail["regime_features"] = features
        else:
            detail["regime"] = REGIME_UNKNOWN
            detail["regime_features"] = {}
        annotated.append(detail)
    return annotated