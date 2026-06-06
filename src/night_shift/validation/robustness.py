"""Robustness gates — Monte Carlo DD, CPCV/PBO, param sensitivity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from night_shift.core.logging import log
from night_shift.core.types import CandidateResult
from night_shift.models.base import DesignModel
from night_shift.simulation.tokenomics_sim import compute_daily_holder_returns
from night_shift.validation.evaluate import evaluate_on_fold
from night_shift.validation.folds import Fold
from night_shift.validation.gates import RobustnessGate


@dataclass
class MonteCarloResult:
    n_simulations: int
    n_periods: int
    position_pct: float
    initial_capital: float
    observed_dd: float
    dd_p50: float
    dd_p75: float
    dd_p90: float
    dd_p95: float
    dd_p99: float
    dd_worst: float
    prob_dd_gt_10: float
    prob_dd_gt_20: float
    prob_dd_gt_30: float
    prob_dd_gt_50: float
    return_p50: float
    return_p90: float
    return_p10: float
    period_returns: List[float] = field(default_factory=list)
    dd_distribution: List[float] = field(default_factory=list)


@dataclass
class CPCVResult:
    n_folds: int
    n_test_folds: int
    n_paths: int
    pbo: float
    logits: List[float]
    oos_score_distribution: List[float]
    is_score_distribution: List[float]
    path_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SensitivityResult:
    max_sensitivity: float
    fragile_params: List[str]
    perturbations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RobustnessVerdict:
    overall: str
    passed: bool
    flags: List[str] = field(default_factory=list)
    gate_failures: List[str] = field(default_factory=list)


def _simulate_equity(pnls: np.ndarray, position_pct: float, initial_capital: float) -> float:
    capital = initial_capital
    for pnl_pct in pnls:
        position = capital * position_pct
        capital += position * (pnl_pct / 100.0)
        if capital <= 0:
            return 0.0
    return capital


def _compute_max_dd(pnls: np.ndarray, position_pct: float, initial_capital: float) -> float:
    capital = initial_capital
    peak = capital
    max_dd = 0.0
    for pnl_pct in pnls:
        position = capital * position_pct
        capital += position * (pnl_pct / 100.0)
        if capital <= 0:
            return 100.0
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak * 100
        max_dd = max(max_dd, dd)
    return max_dd


def monte_carlo_dd(
    period_returns: List[float],
    position_pct: float = 1.0,
    initial_capital: float = 100.0,
    n_simulations: int = 10000,
) -> MonteCarloResult:
    """Shuffle daily holder-return series and build a drawdown distribution."""
    if not period_returns:
        return MonteCarloResult(
            n_simulations=0,
            n_periods=0,
            position_pct=position_pct,
            initial_capital=initial_capital,
            observed_dd=0,
            dd_p50=0,
            dd_p75=0,
            dd_p90=0,
            dd_p95=0,
            dd_p99=0,
            dd_worst=0,
            prob_dd_gt_10=0,
            prob_dd_gt_20=0,
            prob_dd_gt_30=0,
            prob_dd_gt_50=0,
            return_p50=0,
            return_p90=0,
            return_p10=0,
        )

    pnls = np.array(period_returns, dtype=np.float64)
    observed_dd = _compute_max_dd(pnls, position_pct, initial_capital)

    rng = np.random.default_rng(42)
    dd_dist = np.empty(n_simulations)
    return_dist = np.empty(n_simulations)

    for i in range(n_simulations):
        shuffled = rng.permutation(pnls)
        dd_dist[i] = _compute_max_dd(shuffled, position_pct, initial_capital)
        final_cap = _simulate_equity(shuffled, position_pct, initial_capital)
        return_dist[i] = (final_cap - initial_capital) / initial_capital * 100

    dd_sorted = np.sort(dd_dist)
    return MonteCarloResult(
        n_simulations=n_simulations,
        n_periods=len(pnls),
        position_pct=position_pct,
        initial_capital=initial_capital,
        observed_dd=round(float(observed_dd), 2),
        dd_p50=round(float(np.percentile(dd_dist, 50)), 2),
        dd_p75=round(float(np.percentile(dd_dist, 75)), 2),
        dd_p90=round(float(np.percentile(dd_dist, 90)), 2),
        dd_p95=round(float(np.percentile(dd_dist, 95)), 2),
        dd_p99=round(float(np.percentile(dd_dist, 99)), 2),
        dd_worst=round(float(dd_sorted[-1]), 2),
        prob_dd_gt_10=round(float(np.mean(dd_dist > 10)), 4),
        prob_dd_gt_20=round(float(np.mean(dd_dist > 20)), 4),
        prob_dd_gt_30=round(float(np.mean(dd_dist > 30)), 4),
        prob_dd_gt_50=round(float(np.mean(dd_dist > 50)), 4),
        return_p50=round(float(np.percentile(return_dist, 50)), 2),
        return_p90=round(float(np.percentile(return_dist, 90)), 2),
        return_p10=round(float(np.percentile(return_dist, 10)), 2),
        period_returns=list(period_returns),
        dd_distribution=dd_dist.tolist(),
    )


def extract_daily_returns(
    events: pd.DataFrame,
    params: Dict[str, Any],
    folds: List[Fold],
) -> List[float]:
    """Collect per-day holder returns across all walk-forward test windows."""
    returns: List[float] = []
    for fold in folds:
        window = events.iloc[fold.test_start_idx : fold.test_end_idx]
        if len(window) < 2:
            continue
        daily, _ = compute_daily_holder_returns(window, params)
        returns.extend(float(x) for x in daily)
    return returns


def generate_cpcv_param_grid(params: Dict[str, Any], n_variants: int = 20) -> List[Dict[str, Any]]:
    """Perturb numeric tokenomics params for CPCV overfitting detection."""
    variants = [dict(params)]
    rng = np.random.default_rng(42)
    skip = {"vesting_type", "treasury_control"}
    numeric_keys = [
        k for k, v in params.items() if isinstance(v, (int, float)) and k not in skip
    ]

    for _ in range(max(0, n_variants - 1)):
        variant = dict(params)
        if not numeric_keys:
            break
        n_perturb = int(rng.integers(2, min(4, len(numeric_keys) + 1)))
        keys = rng.choice(numeric_keys, size=n_perturb, replace=False)
        for key in keys:
            delta = rng.uniform(0.10, 0.20) * rng.choice([-1, 1])
            original = variant[key]
            if isinstance(original, int):
                variant[key] = max(0, int(original * (1 + delta)))
            else:
                variant[key] = max(0.01, round(original * (1 + delta), 4))
        variants.append(variant)
    return variants


def cpcv(
    events: pd.DataFrame,
    model: DesignModel,
    params_grid: List[Dict[str, Any]],
    folds: List[Fold],
    n_test_folds: int = 2,
) -> CPCVResult:
    """Combinatorial purged cross-validation with PBO for tokenomic designs."""
    if len(folds) < n_test_folds:
        return CPCVResult(
            n_folds=len(folds),
            n_test_folds=n_test_folds,
            n_paths=0,
            pbo=1.0,
            logits=[],
            oos_score_distribution=[],
            is_score_distribution=[],
        )

    fold_indices = list(range(len(folds)))
    test_combos = list(combinations(fold_indices, n_test_folds))
    logits: List[float] = []
    oos_scores: List[float] = []
    is_scores: List[float] = []
    path_results: List[Dict[str, Any]] = []

    for path_idx, test_indices in enumerate(test_combos):
        train_indices = tuple(i for i in fold_indices if i not in test_indices)
        train_folds = [folds[i] for i in train_indices]

        best_is = -999.0
        best_params: Optional[Dict[str, Any]] = None
        for params in params_grid:
            scores = []
            for fold in train_folds:
                result = evaluate_on_fold(events, fold, params, model, skip_is=True)
                scores.append(result["oos_score"])
            mean_is = float(np.median(scores)) if scores else 0.0
            if mean_is > best_is:
                best_is = mean_is
                best_params = params

        if best_params is None:
            logits.append(0.0)
            continue

        oos_path = []
        for fi in test_indices:
            result = evaluate_on_fold(events, folds[fi], best_params, model, skip_is=True)
            oos_path.append(result["oos_score"])
        median_oos = float(np.median(oos_path)) if oos_path else 0.0

        all_test_scores = []
        for params in params_grid:
            p_scores = []
            for fi in test_indices:
                result = evaluate_on_fold(events, folds[fi], params, model, skip_is=True)
                p_scores.append(result["oos_score"])
            all_test_scores.append(float(np.median(p_scores)) if p_scores else 0.0)

        rank = sum(1 for s in all_test_scores if s <= median_oos)
        n_params = len(all_test_scores)
        denominator = n_params - rank + 1
        logit = float(np.log(rank / denominator)) if denominator > 0 and rank > 0 else -5.0

        logits.append(round(logit, 4))
        oos_scores.append(round(median_oos, 4))
        is_scores.append(round(best_is, 4))
        path_results.append(
            {
                "path": path_idx,
                "test_folds": list(test_indices),
                "train_folds": list(train_indices),
                "best_is_score": round(best_is, 4),
                "oos_score": round(median_oos, 4),
                "rank": rank,
                "n_params": n_params,
                "logit": round(logit, 4),
            }
        )

    pbo = sum(1 for l in logits if l < 0) / len(logits) if logits else 1.0
    return CPCVResult(
        n_folds=len(folds),
        n_test_folds=n_test_folds,
        n_paths=len(test_combos),
        pbo=round(pbo, 4),
        logits=logits,
        oos_score_distribution=oos_scores,
        is_score_distribution=is_scores,
        path_results=path_results,
    )


def param_sensitivity(
    events: pd.DataFrame,
    model: DesignModel,
    params: Dict[str, Any],
    folds: List[Fold],
    perturbation: float = 0.10,
) -> SensitivityResult:
    """Measure score sensitivity to ±10% parameter perturbations (fragility gate)."""
    if not folds:
        return SensitivityResult(max_sensitivity=0.0, fragile_params=[])

    baseline = evaluate_on_fold(events, folds[-1], params, model, skip_is=True)
    base_score = baseline["oos_score"]
    if abs(base_score) < 0.01:
        return SensitivityResult(max_sensitivity=0.0, fragile_params=[])

    max_sensitivity = 0.0
    fragile: List[str] = []
    perturbations: List[Dict[str, Any]] = []
    validated = model.validate_params(params)

    for param_name, param_val in validated.items():
        if model.is_discrete(param_name) or not isinstance(param_val, (int, float)):
            continue
        for delta in (-perturbation, perturbation):
            perturbed = dict(validated)
            floor = model.param_floor(param_name)
            if isinstance(param_val, int):
                perturbed[param_name] = max(int(floor), int(param_val * (1 + delta)))
            else:
                perturbed[param_name] = max(floor, round(param_val * (1 + delta), 4))
            result = evaluate_on_fold(events, folds[-1], perturbed, model, skip_is=True)
            sensitivity = abs(result["oos_score"] - base_score) / abs(base_score)
            max_sensitivity = max(max_sensitivity, sensitivity)
            perturbations.append(
                {
                    "param": param_name,
                    "delta": delta,
                    "oos_score": result["oos_score"],
                    "sensitivity": round(sensitivity, 4),
                }
            )
            if sensitivity > 0.4:
                fragile.append(param_name)

    return SensitivityResult(
        max_sensitivity=round(max_sensitivity, 4),
        fragile_params=sorted(set(fragile)),
        perturbations=perturbations,
    )


def _build_verdict(
    mc: MonteCarloResult,
    cpcv: CPCVResult,
    sensitivity: SensitivityResult,
    gate: RobustnessGate,
) -> RobustnessVerdict:
    flags: List[str] = []
    gate_failures: List[str] = []

    if mc.dd_p95 > gate.MAX_MC_DD_P95:
        msg = f"MC DD p95={mc.dd_p95:.1f}% exceeds {gate.MAX_MC_DD_P95}%"
        flags.append(msg)
        gate_failures.append(msg)
    if mc.prob_dd_gt_30 > gate.MAX_PROB_DD_GT_30:
        msg = f"P(DD>30%)={mc.prob_dd_gt_30:.1%} exceeds {gate.MAX_PROB_DD_GT_30:.0%}"
        flags.append(msg)
        gate_failures.append(msg)
    if cpcv.pbo > gate.MAX_PBO:
        msg = f"PBO={cpcv.pbo:.0%} indicates likely overfitting"
        flags.append(msg)
        gate_failures.append(msg)
    elif cpcv.pbo > gate.CAUTION_PBO:
        flags.append(f"PBO={cpcv.pbo:.0%} is elevated (>{gate.CAUTION_PBO:.0%})")
    if sensitivity.max_sensitivity > gate.MAX_PARAM_SENSITIVITY:
        msg = (
            f"Param sensitivity={sensitivity.max_sensitivity:.2f} "
            f"exceeds {gate.MAX_PARAM_SENSITIVITY}"
        )
        flags.append(msg)
        gate_failures.append(msg)
    if mc.return_p10 < gate.MIN_RETURN_P10:
        msg = f"Return p10={mc.return_p10:.1f}% below floor"
        flags.append(msg)
        gate_failures.append(msg)

    passed = len(gate_failures) == 0
    if passed and not flags:
        overall = "PASS — Design passes robustness gates"
    elif passed:
        overall = "CAUTION — Minor robustness concerns"
    elif len(gate_failures) == 1:
        overall = "FAIL — One robustness gate failed"
    else:
        overall = "FAIL — Multiple robustness gates failed"

    return RobustnessVerdict(overall=overall, passed=passed, flags=flags, gate_failures=gate_failures)


def run_robustness_analysis(
    token: str,
    events: pd.DataFrame,
    params: Dict[str, Any],
    model: DesignModel,
    folds: List[Fold],
    *,
    n_mc_simulations: int = 1000,
    n_cpcv_variants: int = 12,
    n_cpcv_test_folds: int = 2,
    position_pct: float = 1.0,
    gate: Optional[RobustnessGate] = None,
) -> Dict[str, Any]:
    """Full robustness analysis for one design candidate."""
    gate = gate or RobustnessGate()

    period_returns = extract_daily_returns(events, params, folds)
    mc_result = monte_carlo_dd(
        period_returns,
        position_pct=position_pct,
        n_simulations=n_mc_simulations,
    )

    params_grid = generate_cpcv_param_grid(params, n_variants=n_cpcv_variants)
    cpcv_result = cpcv(
        events,
        model,
        params_grid,
        folds,
        n_test_folds=min(n_cpcv_test_folds, max(1, len(folds) - 1)),
    )

    sensitivity_result = param_sensitivity(events, model, params, folds)
    verdict = _build_verdict(mc_result, cpcv_result, sensitivity_result, gate)

    return {
        "token": token,
        "params": params,
        "monte_carlo": asdict(mc_result),
        "cpcv": {
            **asdict(cpcv_result),
            "logit_mean": round(float(np.mean(cpcv_result.logits)), 4) if cpcv_result.logits else 0,
            "logit_median": round(float(np.median(cpcv_result.logits)), 4) if cpcv_result.logits else 0,
        },
        "sensitivity": asdict(sensitivity_result),
        "verdict": asdict(verdict),
    }


def run_robustness_phase(
    token_data: Dict[str, pd.DataFrame],
    all_results: Dict[str, List[CandidateResult]],
    model: DesignModel,
    folds: List[Fold],
    config: Dict[str, Any],
    output_dir: Optional[str | Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run robustness gates on top survivors per token."""
    robustness_config = config.get("robustness", {})
    if not robustness_config.get("enabled", False):
        log("  Robustness gates disabled")
        return {}

    gate = RobustnessGate(
        MAX_MC_DD_P95=robustness_config.get("max_mc_dd_p95", 40.0),
        MAX_PROB_DD_GT_30=robustness_config.get("max_prob_dd_gt_30", 0.25),
        MAX_PBO=robustness_config.get("max_pbo", 0.30),
        CAUTION_PBO=robustness_config.get("caution_pbo", 0.15),
        MAX_PARAM_SENSITIVITY=robustness_config.get("max_param_sensitivity", 0.4),
        MIN_RETURN_P10=robustness_config.get("min_return_p10", -50.0),
    )

    top_n = robustness_config.get("top_n", 3)
    n_mc = robustness_config.get("mc_simulations", 1000)
    n_variants = robustness_config.get("cpcv_variants", 12)
    n_test = robustness_config.get("cpcv_test_folds", 2)
    position_pct = robustness_config.get("position_pct", 1.0)
    reject_on_fail = robustness_config.get("reject_on_fail", True)

    log(f"  Testing top {top_n} survivors per token (MC={n_mc}, CPCV variants={n_variants})")

    all_robustness: Dict[str, List[Dict[str, Any]]] = {}
    save_dir = Path(output_dir) if output_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for token, results in all_results.items():
        events = token_data.get(token)
        if events is None:
            continue

        survivors = sorted(
            [r for r in results if not r.rejected],
            key=lambda r: r.survivor_score,
            reverse=True,
        )[:top_n]

        if not survivors:
            log(f"    [{token}] No survivors for robustness testing")
            continue

        token_results: List[Dict[str, Any]] = []
        for i, candidate in enumerate(survivors, 1):
            log(f"    [{token}] Candidate #{i} survivor={candidate.survivor_score:.3f}")
            analysis = run_robustness_analysis(
                token,
                events,
                candidate.params,
                model,
                folds,
                n_mc_simulations=n_mc,
                n_cpcv_variants=n_variants,
                n_cpcv_test_folds=n_test,
                position_pct=position_pct,
                gate=gate,
            )
            verdict = analysis["verdict"]
            mc = analysis["monte_carlo"]
            cpcv = analysis["cpcv"]
            log(
                f"      MC DD p95={mc['dd_p95']:.1f}% | PBO={cpcv['pbo']:.1%} | "
                f"Sensitivity={analysis['sensitivity']['max_sensitivity']:.2f} | "
                f"{verdict['overall']}"
            )

            if reject_on_fail and not verdict["passed"]:
                candidate.rejected = True
                candidate.rejection_reason = "; ".join(verdict["gate_failures"]) or verdict["overall"]

            token_results.append(analysis)

        all_robustness[token] = token_results
        if save_dir:
            out_path = save_dir / f"{token}_robustness.json"
            out_path.write_text(json.dumps(token_results, indent=2, default=str))

    return all_robustness