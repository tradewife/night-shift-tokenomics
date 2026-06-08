"""
Domain-agnostic parameter-space generation and 9-fold expanding-window
walk-forward validation (WFA) harness.

================================================================================
MIGRATION NOTE
================================================================================
During Mission 000 this module lives at:
    research/missions/000-wfa-extraction/output/validation.py

After Hermes validation and human sign-off, relocate to:
    research/engine/validation.py

Intended integration with the original orchestrator (conceptual; file untouched
during Mission 000):

    from research.engine.validation import (
        WFAConfig,
        OverfittingConfig,
        DarwinianConfig,
        Fold,
        FoldOutcome,
        CandidateResult,
        FoldEvaluator,
        create_folds,
        grid_combos,
        evaluate_candidate,
        run_coarse_screen,
        run_fine_refinement,
        darwinian_evolution,
    )

    # 1. Supply a domain-specific fold evaluator (trading, tokenomics, security).
    def my_evaluator(data: Any, fold: Fold, params: Dict[str, Any]) -> FoldOutcome:
        # Run in-sample optimization on fold.train_* indices,
        # evaluate out-of-sample on fold.test_* indices.
        # Return FoldOutcome with is_score and oos_score populated.
        ...

    # 2. Build expanding-window folds from data length.
    wfa = WFAConfig()  # defaults: 9 folds, 36-day test windows
    folds = create_folds(
        total_bars=len(data),
        num_folds=wfa.num_folds,
        test_fold_bars=wfa.test_fold_bars,
        warmup_bars=wfa.warmup_bars,
    )

    # 3. Parameter-space generation + staged search.
    coarse_results = run_coarse_screen(
        data, coarse_param_grid, my_evaluator, overfitting_config=OverfittingConfig()
    )
    top = sorted(coarse_results, key=lambda r: r.survivor_score, reverse=True)[:100]
    fine_results = run_fine_refinement(
        data, folds, top, refinement_grid, my_evaluator,
        overfitting_config=OverfittingConfig(), compute_fragility=True,
    )
    survivors = darwinian_evolution(
        data, folds, fine_results, my_evaluator,
        darwinian_config=DarwinianConfig(),
        overfitting_config=OverfittingConfig(),
    )

Public interface summary:
    - grid_combos: Cartesian product over a parameter grid dict.
    - create_folds: 9-fold (max) expanding-window, non-overlapping test segments.
    - evaluate_candidate: Full WFA harness (aggregation, consistency, survivor score).
    - run_coarse_screen / run_fine_refinement: Two-stage grid search orchestration.
    - darwinian_evolution: Perturbation-based survivor selection over top candidates.

Lineage: extracted from research/orchestration/night_shift.py (Mission 000, 2026-06-09).
Statistical contracts preserved; domain evaluation delegated to FoldEvaluator callback.
================================================================================
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

# Type aliases
ParamDict = Dict[str, Any]
ParamGrid = Dict[str, Sequence[Any]]


# ---------------------------------------------------------------------------
# Configuration dataclasses (statistical contracts)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WFAConfig:
    """Walk-forward validation window contract.

    Statistical assumptions:
        - Up to ``num_folds`` (default 9) non-overlapping test segments partition
          the post-warmup timeline.
        - Each fold uses strictly earlier observations for in-sample configuration
          (train_start..train_end) and strictly later observations for
          out-of-sample validation (test_start..test_end). Train always ends
          where test begins; test segments are contiguous and non-overlapping.
        - Every post-warmup bar appears in exactly one test fold.
        - ``test_fold_bars`` sets the nominal test window size; actual fold count
          may be fewer than ``num_folds`` when data is insufficient.
    """

    num_folds: int = 9
    test_fold_bars: int = 36 * 24  # 36 days at 24 bars/day (original default)
    warmup_bars: int = 250


@dataclass(frozen=True)
class OverfittingConfig:
    """Gates and penalties applied during survivor scoring.

    Statistical assumptions:
        - ``min_fold_consistency``: minimum fraction of folds whose primary
          out-of-sample score exceeds ``consistency_threshold`` (hard rejection).
        - ``max_is_oos_gap``: maximum allowed relative in-sample vs out-of-sample
          degradation (hard rejection).
        - ``min_activity_per_fold``: minimum mean activity count per fold for
          full survivor-score credit (soft penalty below threshold).
        - Fragility is a soft penalty (not a hard rejection): parameter
          sensitivity measured by ±10% perturbations on the final fold.
    """

    max_is_oos_gap: float = 0.5
    min_fold_consistency: float = 0.50
    consistency_threshold: float = 0.0
    min_activity_per_fold: float = 10.0
    score_winsorize_cap: float = 100.0


@dataclass(frozen=True)
class DarwinianConfig:
    """Darwinian survivor-selection contract.

    Statistical assumptions:
        - Seed population: top ``population`` non-rejected candidates by
          survivor_score.
        - Each generation: 3 offspring per parent via random ±perturbation on
          one numeric parameter.
        - Selection: parents + offspring ranked by survivor_score; top
          ``population`` advance.
        - Output: deduplicated union of all generation elites, capped at
          ``population * 2``.
    """

    generations: int = 5
    population: int = 50
    perturbation_range: Tuple[float, float] = (0.05, 0.15)
    offspring_per_parent: int = 3
    perturb_exclude_keys: Tuple[str, ...] = ()
    param_floors: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fold geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One expanding-window train/test split.

    Indices are half-open ``[start, end)`` into a sequential observation array.
    Temporal contract: ``train_end_idx == test_start_idx`` and
    ``train_start_idx < train_end_idx <= test_end_idx``.
    """

    fold_num: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_bars: int
    test_bars: int


def create_folds(
    total_bars: int,
    num_folds: int,
    test_fold_bars: int,
    warmup_bars: int = 250,
) -> List[Fold]:
    """Create non-overlapping expanding-window WFA folds.

    Data layout (expanding train, rolling test):

        Fold 0: [TRAIN====][TEST]
        Fold 1: [TRAIN============][TEST]
        ...

    Statistical contract (preserved from original night_shift.py):
        - ``warmup_bars`` leading observations are reserved; first test window
          starts at ``warmup_bars``.
        - Train window always spans ``[0, test_start)`` — all prior data.
        - Test windows are contiguous, non-overlapping, and collectively cover
          ``[warmup_bars, total_bars)`` without gaps.
        - ``num_folds`` is a maximum; fewer folds are emitted when data cannot
          support the nominal ``test_fold_bars`` window ``num_folds`` times.
        - When ``actual_folds < max_folds``, each test window equals
          ``test_fold_bars`` (no inflation beyond nominal size).
        - When ``actual_folds == max_folds``, usable post-warmup data is divided
          evenly across folds (last fold extends to ``total_bars``).

    Args:
        total_bars: Total observation count in the sequential dataset.
        num_folds: Maximum number of test folds (default contract: 9).
        test_fold_bars: Nominal test-window size in bars.
        warmup_bars: Leading bars excluded from the first test window.

    Returns:
        Ordered list of :class:`Fold` instances, earliest test window first.
    """
    if total_bars <= warmup_bars + test_fold_bars:
        return [
            Fold(
                fold_num=0,
                train_start_idx=0,
                train_end_idx=warmup_bars,
                test_start_idx=warmup_bars,
                test_end_idx=total_bars,
                train_bars=warmup_bars,
                test_bars=total_bars - warmup_bars,
            )
        ]

    usable_bars = total_bars - warmup_bars
    max_folds = usable_bars // test_fold_bars
    actual_folds = min(num_folds, max_folds)

    if actual_folds < max_folds:
        bars_per_fold = test_fold_bars
    else:
        bars_per_fold = usable_bars // actual_folds

    folds: List[Fold] = []
    test_start = warmup_bars
    for i in range(actual_folds):
        test_end = test_start + bars_per_fold if i < actual_folds - 1 else total_bars
        folds.append(
            Fold(
                fold_num=i,
                train_start_idx=0,
                train_end_idx=test_start,
                test_start_idx=test_start,
                test_end_idx=test_end,
                train_bars=test_start,
                test_bars=test_end - test_start,
            )
        )
        test_start = test_end

    return folds


# ---------------------------------------------------------------------------
# Parameter-space generation
# ---------------------------------------------------------------------------

def grid_combos(grid: ParamGrid) -> List[ParamDict]:
    """Generate the Cartesian product of a parameter grid.

    Each key in ``grid`` maps to a sequence of candidate values. Returns every
    combination as an independent parameter dict suitable for evaluation.

    Args:
        grid: Mapping of parameter names to finite value lists.

    Returns:
        List of parameter dicts, one per grid point. Order follows
        ``itertools.product`` over keys in insertion order.
    """
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ---------------------------------------------------------------------------
# Evaluation contracts
# ---------------------------------------------------------------------------

@dataclass
class FoldOutcome:
    """Out-of-sample and in-sample metrics for a single fold.

    Domain evaluators populate these fields; the WFA harness aggregates them.
    Naming is intentionally domain-neutral (no trading, PnL, or asset semantics).

    Fields:
        is_score: In-sample primary metric after training/configuration on
            ``fold.train_*`` indices.
        oos_score: Out-of-sample primary metric on ``fold.test_*`` indices.
            Consistency is computed against ``consistency_threshold`` on this
            field (original: positive OOS Sharpe).
        oos_outcome: Secondary outcome metric (e.g. cumulative effect size).
        oos_risk: Risk metric for survivor penalty (higher = worse).
        oos_activity: Activity/event count in the test window.
    """

    is_score: float = 0.0
    is_outcome: float = 0.0
    oos_score: float = 0.0
    oos_outcome: float = 0.0
    oos_risk: float = 0.0
    oos_activity: float = 0.0


class FoldEvaluator(Protocol):
    """Callable contract for domain-specific per-fold evaluation."""

    def __call__(self, data: Any, fold: Fold, params: ParamDict) -> FoldOutcome:
        """Evaluate ``params`` on one fold.

        Must respect temporal separation: in-sample work uses only
        ``[train_start_idx, train_end_idx)``; out-of-sample scoring uses only
        ``[test_start_idx, test_end_idx)``.
        """
        ...


@dataclass
class CandidateResult:
    """Full WFA result for one parameter configuration.

    Includes aggregate metrics, consistency, overfitting diagnostics,
    survivor ranking score, per-fold detail, and provenance lineage.
    """

    params: ParamDict
    # Aggregate out-of-sample
    oos_score: float
    oos_outcome: float
    oos_risk: float
    fold_consistency: float
    oos_avg_activity_per_fold: float
    # In-sample
    is_score: float
    is_outcome: float
    # Overfitting
    overfitting_score: float
    fragility: float
    # Ranking
    survivor_score: float
    # Per-fold detail
    folds: List[Dict[str, Any]] = field(default_factory=list)
    # Gates
    rejected: bool = False
    rejection_reason: str = ""
    is_coarse_only: bool = False
    # Lineage
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    label: str = ""


def _winsorize_scores(
    scores: List[float], cap: float
) -> Tuple[List[float], List[float]]:
    """Winsorize per-fold primary scores at ±cap.

    Prevents tiny-sample outliers from dominating aggregates (original contract:
    per-fold Sharpe capped at ±100).
    """
    raw = list(scores)
    clipped = [max(-cap, min(cap, s)) for s in raw]
    return clipped, raw


def _compute_overfitting_score(
    avg_is_score: float,
    avg_oos_score: float,
    is_skipped: bool,
) -> float:
    """Relative in-sample vs out-of-sample degradation."""
    if is_skipped:
        return 0.0
    if abs(avg_is_score) > 0.01:
        gap = (avg_is_score - avg_oos_score) / abs(avg_is_score)
    else:
        gap = 0.5 if avg_oos_score < 0 else 0.0
    return max(0.0, gap)


def _compute_fragility(
    data: Any,
    folds: Sequence[Fold],
    params: ParamDict,
    evaluator: FoldEvaluator,
    avg_oos_score: float,
    score_cap: float,
    perturb_exclude: Sequence[str] = (),
) -> float:
    """Parameter sensitivity via ±10% perturbations on the final fold.

    Returns maximum relative primary-score change across perturbations.
    """
    if not folds or avg_oos_score <= 0.1:
        return 0.0

    fragility = 0.0
    last_fold = folds[-1]
    for param_name, param_val in params.items():
        if param_name in perturb_exclude:
            continue
        if not isinstance(param_val, (int, float)):
            continue
        for delta in (-0.10, 0.10):
            perturbed = {**params, param_name: round(param_val * (1 + delta), 4)}
            outcome = evaluator(data, last_fold, perturbed)
            perturbed_score = max(
                -score_cap, min(score_cap, outcome.oos_score)
            )
            if abs(avg_oos_score) > 0.01:
                sensitivity = abs(perturbed_score - avg_oos_score) / abs(
                    avg_oos_score
                )
                fragility = max(fragility, sensitivity)
    return fragility


def _compute_survivor_score(
    avg_oos_score: float,
    fold_consistency: float,
    overfitting_score: float,
    avg_oos_risk: float,
    avg_oos_activity: float,
    fragility: float,
    of_config: OverfittingConfig,
) -> float:
    """Darwinian survivor ranking formula (preserved from original).

    survivor = median_oos_score
             × fold_consistency
             × (1 - min(overfitting, 1))
             × 1/(1 + risk/100)
             × min(activity / min_activity, 1)
             × 1/(1 + fragility)
    """
    of_penalty = 1.0 - min(overfitting_score, 1.0)
    risk_factor = 1.0 / (1.0 + avg_oos_risk / 100.0)
    activity_factor = min(
        avg_oos_activity / max(of_config.min_activity_per_fold, 1.0), 1.0
    )
    fragility_penalty = 1.0 / (1.0 + fragility)
    return (
        avg_oos_score
        * fold_consistency
        * of_penalty
        * risk_factor
        * activity_factor
        * fragility_penalty
    )


def evaluate_candidate(
    data: Any,
    folds: Sequence[Fold],
    params: ParamDict,
    evaluator: FoldEvaluator,
    overfitting_config: Optional[OverfittingConfig] = None,
    *,
    compute_fragility: bool = False,
    skip_is: bool = False,
    label: str = "",
) -> CandidateResult:
    """Full WFA evaluation of one parameter configuration.

    Statistical contract (preserved from original night_shift.py):
        1. Evaluate every fold sequentially via ``evaluator``.
        2. Winsorize per-fold primary scores at ±``score_winsorize_cap``.
        3. Aggregate primary score with **median** (robust to single-fold outliers).
        4. **Consistency** = fraction of folds where winsorized ``oos_score`` >
           ``consistency_threshold`` (original: positive OOS Sharpe → threshold 0).
        5. **Overfitting score** = relative IS-OOS gap on primary score.
        6. **Fragility** (optional) = max ±10% parameter sensitivity on final fold.
        7. **Survivor score** = multiplicative ranking formula (see
           :func:`_compute_survivor_score`).
        8. Hard rejection if overfitting gap or consistency violates
           ``OverfittingConfig`` thresholds.
        9. ``is_coarse_only`` when fewer than 3 folds evaluated.

    Args:
        data: Domain dataset (opaque to this harness).
        folds: Expanding-window fold definitions.
        params: Parameter configuration to evaluate.
        evaluator: Domain-specific per-fold evaluator.
        overfitting_config: Gate and penalty thresholds.
        compute_fragility: If True, run parameter-sensitivity perturbations.
        skip_is: If True, evaluator may omit in-sample work (coarse screen pass).
        label: Optional lineage label (e.g. experiment name).

    Returns:
        :class:`CandidateResult` with aggregates, gates, and per-fold detail.
    """
    of_config = overfitting_config or OverfittingConfig()

    fold_outcomes: List[FoldOutcome] = []
    for fold in folds:
        outcome = evaluator(data, fold, params)
        if skip_is:
            outcome = FoldOutcome(
                is_score=0.0,
                is_outcome=0.0,
                oos_score=outcome.oos_score,
                oos_outcome=outcome.oos_outcome,
                oos_risk=outcome.oos_risk,
                oos_activity=outcome.oos_activity,
            )
        fold_outcomes.append(outcome)

    oos_scores_raw = [o.oos_score for o in fold_outcomes]
    oos_scores, oos_scores_unclipped = _winsorize_scores(
        oos_scores_raw, of_config.score_winsorize_cap
    )
    oos_outcomes = [o.oos_outcome for o in fold_outcomes]
    oos_risks = [o.oos_risk for o in fold_outcomes]
    oos_activities = [o.oos_activity for o in fold_outcomes]
    is_scores = [o.is_score for o in fold_outcomes]
    is_outcomes = [o.is_outcome for o in fold_outcomes]

    avg_is_score = float(np.mean(is_scores)) if is_scores else 0.0
    avg_is_outcome = float(np.sum(is_outcomes)) if is_outcomes else 0.0
    avg_oos_score = float(np.median(oos_scores)) if oos_scores else 0.0
    avg_oos_outcome = float(np.sum(oos_outcomes)) if oos_outcomes else 0.0
    avg_oos_risk = float(np.mean(oos_risks)) if oos_risks else 0.0
    avg_oos_activity = float(np.mean(oos_activities)) if oos_activities else 0.0

    meeting_threshold = sum(
        1 for s in oos_scores if s > of_config.consistency_threshold
    )
    fold_consistency = (
        meeting_threshold / len(oos_scores) if oos_scores else 0.0
    )

    overfitting_score = _compute_overfitting_score(
        avg_is_score, avg_oos_score, skip_is
    )
    is_coarse_only = len(fold_outcomes) < 3

    fragility = 0.0
    if compute_fragility:
        fragility = _compute_fragility(
            data,
            folds,
            params,
            evaluator,
            avg_oos_score,
            of_config.score_winsorize_cap,
        )

    survivor_score = _compute_survivor_score(
        avg_oos_score,
        fold_consistency,
        overfitting_score,
        avg_oos_risk,
        avg_oos_activity,
        fragility,
        of_config,
    )

    rejected = False
    rejection_reason = ""
    if overfitting_score > of_config.max_is_oos_gap:
        rejected = True
        rejection_reason = (
            f"overfitting_score={overfitting_score:.2f} > "
            f"{of_config.max_is_oos_gap}"
        )
    if fold_consistency < of_config.min_fold_consistency:
        rejected = True
        rejection_reason = (
            f"fold_consistency={fold_consistency:.0%} < "
            f"{of_config.min_fold_consistency:.0%}"
        )

    fold_details = [
        {
            "fold": folds[i].fold_num if i < len(folds) else i,
            "is_score": fold_outcomes[i].is_score,
            "oos_score": oos_scores[i],
            "oos_score_raw": oos_scores_unclipped[i],
            "oos_outcome": fold_outcomes[i].oos_outcome,
            "oos_activity": fold_outcomes[i].oos_activity,
        }
        for i in range(len(fold_outcomes))
    ]

    return CandidateResult(
        params=dict(params),
        oos_score=avg_oos_score,
        oos_outcome=avg_oos_outcome,
        oos_risk=avg_oos_risk,
        fold_consistency=fold_consistency,
        oos_avg_activity_per_fold=avg_oos_activity,
        is_score=avg_is_score,
        is_outcome=avg_is_outcome,
        overfitting_score=overfitting_score,
        fragility=fragility,
        survivor_score=survivor_score,
        folds=fold_details,
        rejected=rejected,
        rejection_reason=rejection_reason,
        is_coarse_only=is_coarse_only,
        label=label,
    )


# ---------------------------------------------------------------------------
# Staged grid search orchestration
# ---------------------------------------------------------------------------

def run_coarse_screen(
    data: Any,
    param_grid: ParamGrid,
    evaluator: FoldEvaluator,
    overfitting_config: Optional[OverfittingConfig] = None,
    *,
    screen_window_bars: int = 720,
    warmup_bars: int = 250,
    base_params: Optional[ParamDict] = None,
    full_folds: Optional[Sequence[Fold]] = None,
) -> List[CandidateResult]:
    """Stage 1: Fast single-window screen over a coarse parameter grid.

    Evaluates each combination on one truncated test window for rough ordering.
    Skips in-sample evaluation (``skip_is=True``) and fragility. Full WFA on
    all folds happens in :func:`run_fine_refinement`.

    Args:
        data: Domain dataset; ``len(data)`` used when ``full_folds`` absent.
        param_grid: Coarse parameter grid.
        evaluator: Domain fold evaluator.
        overfitting_config: Gate thresholds (consistency still computed on OOS).
        screen_window_bars: Test-window size for the coarse screen (original: 720).
        warmup_bars: Warmup reserved before the screen test window.
        base_params: Fixed parameters merged into every combo.
        full_folds: Optional pre-built fold set; last fold used when data is short.

    Returns:
        One :class:`CandidateResult` per grid combination.
    """
    of_config = overfitting_config or OverfittingConfig()
    combos = grid_combos(param_grid)
    total_bars = len(data)

    if total_bars > screen_window_bars:
        coarse_fold = Fold(
            fold_num=0,
            train_start_idx=0,
            train_end_idx=max(0, total_bars - screen_window_bars - warmup_bars),
            test_start_idx=total_bars - screen_window_bars,
            test_end_idx=total_bars,
            train_bars=max(0, total_bars - screen_window_bars - warmup_bars),
            test_bars=screen_window_bars,
        )
    elif full_folds:
        coarse_fold = full_folds[-1]
    else:
        coarse_fold = create_folds(total_bars, 1, screen_window_bars, warmup_bars)[
            -1
        ]

    results: List[CandidateResult] = []
    for combo in combos:
        params = {**(base_params or {}), **combo}
        results.append(
            evaluate_candidate(
                data,
                [coarse_fold],
                params,
                evaluator,
                of_config,
                compute_fragility=False,
                skip_is=True,
            )
        )
    return results


def run_fine_refinement(
    data: Any,
    folds: Sequence[Fold],
    top_candidates: Sequence[CandidateResult],
    refinement_grid: ParamGrid,
    evaluator: FoldEvaluator,
    overfitting_config: Optional[OverfittingConfig] = None,
    *,
    compute_fragility: bool = True,
    strip_keys: Optional[Sequence[str]] = None,
) -> List[CandidateResult]:
    """Stage 2: Full WFA on all folds for top coarse candidates.

    Re-evaluates each parent configuration with refinement-grid sweeps over
    selected parameters (original: trailing/flip/leverage re-sweep).

    Args:
        data: Domain dataset.
        folds: Complete expanding-window fold set (typically 9 folds).
        top_candidates: Survivors from coarse screen, ranked by survivor_score.
        refinement_grid: Fine-grained parameter grid for re-sweep keys.
        evaluator: Domain fold evaluator.
        overfitting_config: Gate thresholds.
        compute_fragility: Enable parameter-sensitivity penalty.
        strip_keys: Parameter keys removed from parent before refinement merge
            (re-swept via ``refinement_grid``).

    Returns:
        Deduplicated fine-evaluated candidate results.
    """
    of_config = overfitting_config or OverfittingConfig()
    strip = set(strip_keys or ())
    fine_combos = grid_combos(refinement_grid)

    results: List[CandidateResult] = []
    seen_keys: set = set()

    for parent in top_candidates:
        if parent.rejected:
            continue
        base = dict(parent.params)
        for key in strip:
            base.pop(key, None)

        for combo in fine_combos:
            params = {**base, **combo}
            key = tuple(sorted(params.items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append(
                evaluate_candidate(
                    data,
                    folds,
                    params,
                    evaluator,
                    of_config,
                    compute_fragility=compute_fragility,
                )
            )

    return results


def _perturb_params(
    params: ParamDict,
    config: DarwinianConfig,
) -> Optional[ParamDict]:
    """Apply one random multiplicative perturbation to a numeric parameter."""
    mutated = dict(params)
    numeric_keys = [
        k
        for k, v in mutated.items()
        if isinstance(v, (int, float)) and k not in config.perturb_exclude_keys
    ]
    if not numeric_keys:
        return None

    key = random.choice(numeric_keys)
    delta = random.uniform(*config.perturbation_range) * random.choice([-1, 1])
    original = mutated[key]
    floor = config.param_floors.get(key, 0.01)

    if isinstance(original, int):
        mutated[key] = max(1, int(original * (1 + delta)))
    else:
        mutated[key] = max(floor, round(original * (1 + delta), 4))

    return mutated


def darwinian_evolution(
    data: Any,
    folds: Sequence[Fold],
    population: Sequence[CandidateResult],
    evaluator: FoldEvaluator,
    darwinian_config: Optional[DarwinianConfig] = None,
    overfitting_config: Optional[OverfittingConfig] = None,
) -> List[CandidateResult]:
    """Stage 3: Darwinian refinement with random parameter perturbations.

    Statistical contract (preserved from original):
        - Seed with top ``population`` non-rejected candidates by survivor_score.
        - Each generation produces ``offspring_per_parent`` children per parent
          via single-parameter random perturbation.
        - Survivors selected by survivor_score across parents + offspring.
        - Returns deduplicated elites from all generations (cap: population × 2).

    Args:
        data: Domain dataset.
        folds: Full WFA fold set.
        population: Candidate pool (typically fine-refinement output).
        evaluator: Domain fold evaluator.
        darwinian_config: Generation count, population size, perturbation range.
        overfitting_config: Gate thresholds for offspring evaluation.

    Returns:
        Deduplicated survivor list ranked by survivor_score.
    """
    d_config = darwinian_config or DarwinianConfig()
    of_config = overfitting_config or OverfittingConfig()

    current_gen = sorted(
        [r for r in population if not r.rejected],
        key=lambda r: r.survivor_score,
        reverse=True,
    )[: d_config.population]

    if not current_gen:
        return []

    all_survivors: List[CandidateResult] = list(current_gen)

    for _ in range(d_config.generations):
        offspring: List[CandidateResult] = []
        for parent in current_gen:
            for _ in range(d_config.offspring_per_parent):
                params = _perturb_params(parent.params, d_config)
                if params is None:
                    continue
                offspring.append(
                    evaluate_candidate(
                        data,
                        folds,
                        params,
                        evaluator,
                        of_config,
                        compute_fragility=True,
                    )
                )

        combined = current_gen + offspring
        combined.sort(key=lambda r: r.survivor_score, reverse=True)
        current_gen = combined[: d_config.population]
        all_survivors.extend(current_gen)

    seen: set = set()
    unique: List[CandidateResult] = []
    for result in sorted(
        all_survivors, key=lambda r: r.survivor_score, reverse=True
    ):
        key = tuple(sorted(result.params.items()))
        if key not in seen:
            seen.add(key)
            unique.append(result)

    return unique[: d_config.population * 2]


# ---------------------------------------------------------------------------
# Convenience: default WFA fold builder
# ---------------------------------------------------------------------------

def default_folds(total_bars: int, bars_per_day: int = 24) -> List[Fold]:
    """Build folds using :class:`WFAConfig` defaults (9 × 36-day test windows).

    Args:
        total_bars: Total observation count.
        bars_per_day: Bar frequency for converting day-based config to bars.

    Returns:
        Fold list from :func:`create_folds` with standard Mission 000 contract.
    """
    cfg = WFAConfig(test_fold_bars=36 * bars_per_day)
    return create_folds(
        total_bars,
        cfg.num_folds,
        cfg.test_fold_bars,
        cfg.warmup_bars,
    )


__all__ = [
    "WFAConfig",
    "OverfittingConfig",
    "DarwinianConfig",
    "Fold",
    "FoldOutcome",
    "CandidateResult",
    "FoldEvaluator",
    "create_folds",
    "default_folds",
    "grid_combos",
    "evaluate_candidate",
    "run_coarse_screen",
    "run_fine_refinement",
    "darwinian_evolution",
]