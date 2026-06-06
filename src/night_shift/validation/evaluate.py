"""Walk-forward candidate evaluation with injected design models."""

from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from night_shift.core.types import CandidateResult
from night_shift.models.base import DesignModel, WindowMetrics
from night_shift.scoring.fitness import (
    compute_overfitting_score,
    compute_survivor_score,
    winsorize_scores,
)
from night_shift.validation.folds import Fold
from night_shift.validation.gates import RegimeGate
from night_shift.validation.regimes import annotate_fold_regimes, check_regime_gate


def _regime_gate_from_config(of_config: Dict[str, Any]) -> RegimeGate:
    regime_cfg = of_config.get("regime_gate", {})
    return RegimeGate(
        ENABLED=regime_cfg.get("enabled", True),
        MIN_PROFITABLE_REGIMES=regime_cfg.get("min_profitable_regimes", 3),
        MIN_REGIME_SCORE=regime_cfg.get("min_regime_score", 0.0),
    )


def _append_rejection(existing: str, new: str) -> str:
    if not new:
        return existing
    if not existing:
        return new
    return f"{existing}; {new}"


def _metrics_to_dict(is_metrics: WindowMetrics, oos_metrics: WindowMetrics) -> Dict[str, Any]:
    return {
        "is_score": is_metrics.score,
        "is_pnl": is_metrics.pnl,
        "oos_score": oos_metrics.score,
        "oos_pnl": oos_metrics.pnl,
        "oos_pf": oos_metrics.profit_factor,
        "oos_wr": oos_metrics.win_rate,
        "oos_max_dd": oos_metrics.max_drawdown_pct,
        "oos_events": oos_metrics.event_count,
        "oos_avg_duration": oos_metrics.avg_duration,
        "oos_outcomes": oos_metrics.outcomes,
    }


def evaluate_on_fold(
    events: pd.DataFrame,
    fold: Fold,
    params: Dict[str, Any],
    model: DesignModel,
    skip_is: bool = False,
) -> Dict[str, Any]:
    """Evaluate a design on one train/test fold."""
    validated = model.validate_params(params)

    if not skip_is and fold.train_end_idx > fold.train_start_idx:
        is_metrics = model.simulate_window(
            events, validated, fold.train_start_idx, fold.train_end_idx
        )
    else:
        is_metrics = WindowMetrics(0, 0, 0, 0, 0, 0, 0, {})

    oos_metrics = model.simulate_window(
        events, validated, fold.test_start_idx, fold.test_end_idx
    )
    return _metrics_to_dict(is_metrics, oos_metrics)


def evaluate_candidate(
    events: pd.DataFrame,
    folds: List[Fold],
    params: Dict[str, Any],
    token: str,
    model: DesignModel,
    of_config: Dict[str, Any],
    compute_fragility: bool = False,
    skip_is: bool = False,
) -> CandidateResult:
    """Full walk-forward evaluation of one design candidate."""
    fold_results = [evaluate_on_fold(events, fold, params, model, skip_is=skip_is) for fold in folds]

    oos_scores_raw = [f["oos_score"] for f in fold_results]
    oos_scores = winsorize_scores(oos_scores_raw)
    oos_pnls = [f["oos_pnl"] for f in fold_results]
    oos_pfs = [f["oos_pf"] for f in fold_results if f["oos_pf"] < 999]
    oos_wrs = [f["oos_wr"] for f in fold_results]
    oos_dds = [f["oos_max_dd"] for f in fold_results]
    oos_events = [f["oos_events"] for f in fold_results]
    oos_durations = [f["oos_avg_duration"] for f in fold_results]

    is_scores = [f["is_score"] for f in fold_results]
    is_pnls = [f["is_pnl"] for f in fold_results]

    avg_is_score = float(np.mean(is_scores)) if is_scores else 0.0
    avg_is_pnl = float(np.sum(is_pnls)) if is_pnls else 0.0
    avg_oos_score = float(np.median(oos_scores)) if oos_scores else 0.0
    avg_oos_pnl = float(np.sum(oos_pnls)) if oos_pnls else 0.0
    avg_oos_pf = float(np.mean(oos_pfs)) if oos_pfs else 0.0
    avg_oos_wr = float(np.mean(oos_wrs)) if oos_wrs else 0.0
    avg_oos_dd = float(np.mean(oos_dds)) if oos_dds else 0.0
    avg_oos_events = float(np.mean(oos_events)) if oos_events else 0.0
    avg_oos_duration = float(np.mean(oos_durations)) if oos_durations else 0.0

    positive_folds = sum(1 for s in oos_scores if s > 0)
    oos_consistency = positive_folds / len(oos_scores) if oos_scores else 0.0

    all_outcomes: Counter = Counter()
    for fold in fold_results:
        for reason, count in fold["oos_outcomes"].items():
            all_outcomes[reason] += count

    overfitting_score = compute_overfitting_score(avg_is_score, avg_oos_score, skip_is)

    fragility = 0.0
    if compute_fragility and avg_oos_score > 0.1 and folds:
        validated = model.validate_params(params)
        for param_name, param_val in validated.items():
            if model.is_discrete(param_name) or not isinstance(param_val, (int, float)):
                continue
            for delta in (-0.10, 0.10):
                perturbed = dict(validated)
                floor = model.param_floor(param_name)
                if isinstance(param_val, int):
                    perturbed[param_name] = max(int(floor), int(param_val * (1 + delta)))
                else:
                    perturbed[param_name] = max(floor, round(param_val * (1 + delta), 4))
                perturbed_result = evaluate_on_fold(events, folds[-1], perturbed, model)
                perturbed_score = max(
                    -100.0,
                    min(100.0, perturbed_result["oos_score"]),
                )
                if abs(avg_oos_score) > 0.01:
                    sensitivity = abs(perturbed_score - avg_oos_score) / abs(avg_oos_score)
                    fragility = max(fragility, sensitivity)

    survivor_score = compute_survivor_score(
        avg_oos_score,
        oos_consistency,
        overfitting_score,
        avg_oos_dd,
        avg_oos_events,
        fragility,
        of_config,
    )

    rejected = False
    rejection_reason = ""
    if overfitting_score > of_config.get("max_is_oos_gap", 0.5):
        rejected = True
        rejection_reason = _append_rejection(
            rejection_reason,
            f"overfitting_score={overfitting_score:.2f} > {of_config.get('max_is_oos_gap', 0.5)}",
        )
    if oos_consistency < of_config.get("min_oos_consistency", 0.50):
        rejected = True
        rejection_reason = _append_rejection(
            rejection_reason,
            f"oos_consistency={oos_consistency:.0%} < "
            f"{of_config.get('min_oos_consistency', 0.50):.0%}",
        )

    fold_details = [
        {
            "fold": folds[i].fold_num if i < len(folds) else i,
            "is_score": f["is_score"],
            "oos_score": oos_scores[i],
            "oos_score_raw": oos_scores_raw[i],
            "oos_pnl": f["oos_pnl"],
            "oos_events": f["oos_events"],
        }
        for i, f in enumerate(fold_results)
    ]
    fold_details = annotate_fold_regimes(events, fold_details, folds)

    regime_gate = _regime_gate_from_config(of_config)
    regime_passed, regime_summary = check_regime_gate(fold_details, regime_gate)
    if regime_gate.ENABLED and not regime_passed:
        rejected = True
        for failure in regime_summary.get("failures", []):
            rejection_reason = _append_rejection(rejection_reason, failure)

    return CandidateResult(
        token=token,
        params=dict(model.validate_params(params)),
        oos_score=avg_oos_score,
        oos_pnl=avg_oos_pnl,
        oos_pf=avg_oos_pf,
        oos_wr=avg_oos_wr,
        oos_max_dd=avg_oos_dd,
        oos_consistency=oos_consistency,
        oos_avg_events_per_fold=avg_oos_events,
        oos_mean_duration=avg_oos_duration,
        oos_outcomes=dict(all_outcomes),
        is_score=avg_is_score,
        is_pnl=avg_is_pnl,
        overfitting_score=overfitting_score,
        fragility=fragility,
        survivor_score=survivor_score,
        folds=fold_details,
        regime_summary=regime_summary,
        rejected=rejected,
        rejection_reason=rejection_reason,
        is_coarse_only=len(fold_results) < 3,
    )