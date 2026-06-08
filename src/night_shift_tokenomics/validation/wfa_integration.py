"""
Lightweight WFA integration for Night Shift Tokenomics.

Designed for economic parameter grid validation (fee structures, emission
schedules, incentive curves, etc.).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from .validation import  # vendored from Mission 000 (
    CandidateResult,
    DarwinianConfig,
    FoldEvaluator,
    FoldOutcome,
    OverfittingConfig,
    WFAConfig,
    create_folds,
    darwinian_evolution,
    evaluate_candidate,
    run_coarse_screen,
    run_fine_refinement,
)


def make_tokenomics_fold_evaluator(
    simulator: Callable[[Any, Any, dict], FoldOutcome]
) -> FoldEvaluator:
    def evaluator(data: Any, fold: Any, params: dict) -> FoldOutcome:
        return simulator(data, fold, params)
    return evaluator


def run_tokenomics_wfa(
    data: Any,
    param_grid: dict[str, list[Any]],
    simulator: Callable[[Any, Any, dict], FoldOutcome],
    wfa_config: Optional[WFAConfig] = None,
    overfitting_config: Optional[OverfittingConfig] = None,
) -> List[CandidateResult]:
    """
    High-level entry point for tokenomics parameter validation.

    Usage example:
        results = run_tokenomics_wfa(
            simulation_data,
            {"fee_rate": [0.001, 0.003, 0.005], "vesting_cliff_days": [...]},
            tokenomics_simulator
        )
    """
    cfg = wfa_config or WFAConfig()
    of_cfg = overfitting_config or OverfittingConfig()
    evaluator = make_tokenomics_fold_evaluator(simulator)

    folds = create_folds(
        total_bars=len(data) if hasattr(data, "__len__") else 10_000,
        num_folds=cfg.num_folds,
        test_fold_bars=cfg.test_fold_bars,
        warmup_bars=cfg.warmup_bars,
    )

    coarse = run_coarse_screen(
        data, param_grid, evaluator, of_cfg, screen_window_bars=720
    )

    top = sorted([c for c in coarse if not c.rejected],
                 key=lambda r: r.survivor_score, reverse=True)[:60]

    fine = run_fine_refinement(
        data, folds, top, {}, evaluator, of_cfg, compute_fragility=True
    )

    survivors = darwinian_evolution(
        data, folds, fine, evaluator, DarwinianConfig(), of_cfg
    )

    return [s for s in survivors if not s.rejected]
