"""Five-stage tokenomics research pipeline orchestrator."""

import time
from typing import Dict, List, Optional

import pandas as pd

from night_shift.config.loader import load_config
from night_shift.core.logging import log
from night_shift.core.types import CandidateResult
from night_shift.data.loader import load_token_data
from night_shift.models.registry import get_model
from night_shift.reporting.report import generate_report
from night_shift.search.darwinian import darwinian_evolution
from night_shift.search.grid import coarse_grid_search, fine_refinement, run_experiments
from night_shift.validation.evaluate import evaluate_candidate
from night_shift.validation.folds import create_folds


def run_night_shift(
    config_path: Optional[str] = None,
    tokens: Optional[List[str]] = None,
    dry_run: Optional[bool] = None,
) -> Dict:
    """
    Main entry point for the Night Shift Tokenomics research pipeline.

    Stages:
      1. Data load
      2. Walk-forward fold creation
      3. Coarse grid + fine refinement
      4. Darwinian evolution
      5. Morning report
    """
    start_time = time.time()
    config = load_config(config_path)

    token_list = tokens or config.get("tokens", ["SYNTHETIC-ALPHA"])
    is_dry_run = dry_run if dry_run is not None else config.get("dry_run", True)
    wfa_config = config.get("wfa", {})
    of_config = config.get("overfitting", {})
    grid_config = config.get("grid_search", {})
    model = get_model(config.get("model", "tokenomics_mvp"))

    log("=" * 70)
    log("NIGHT SHIFT TOKENOMICS — Research Pipeline")
    log(f"Tokens: {', '.join(token_list)}")
    log(f"Model: {model.name}")
    log(f"Mode: {'dry run' if is_dry_run else 'live'}")
    log("=" * 70)

    # Stage 1: Data
    log("\n── Stage 1: Data ──")
    token_data = load_token_data(token_list, dry_run=is_dry_run)
    if not token_data:
        raise RuntimeError("No token data loaded")

    # Stage 2: Walk-forward folds
    log("\n── Stage 2: Walk-Forward Folds ──")
    min_bars = min(len(df) for df in token_data.values())
    folds = create_folds(
        total_bars=min_bars,
        num_folds=wfa_config.get("num_folds", 6),
        test_fold_days=wfa_config.get("test_fold_days", 30),
        warmup_bars=wfa_config.get("warmup_bars", 30),
        bars_per_day=wfa_config.get("bars_per_day", 1),
    )
    log(f"Created {len(folds)} folds from {min_bars} bars")
    for fold in folds:
        log(
            f"  Fold {fold.fold_num}: train=[{fold.train_start_idx}:{fold.train_end_idx}] "
            f"test=[{fold.test_start_idx}:{fold.test_end_idx}]"
        )

    all_results: Dict[str, List[CandidateResult]] = {}

    # Stage 2b: Baseline
    log("\n── Stage 2b: Default Design Baseline ──")
    for token, events in token_data.items():
        baseline = evaluate_candidate(
            events,
            folds,
            model.default_params(),
            token,
            model,
            of_config,
            compute_fragility=True,
        )
        all_results[token] = [baseline]
        log(
            f"  {token}: OOS score={baseline.oos_score:+.3f} "
            f"consistency={baseline.oos_consistency:.0%} "
            f"survivor={baseline.survivor_score:.3f}"
        )

    # Stage 3: Coarse grid
    if grid_config.get("coarse", True):
        log("\n── Stage 3: Coarse Grid Search ──")
        for token, events in token_data.items():
            coarse_results = coarse_grid_search(
                events,
                folds,
                token,
                model,
                of_config,
                coarse_window_days=grid_config.get("coarse_window_days", 30),
                bars_per_day=wfa_config.get("bars_per_day", 1),
            )
            all_results[token].extend(coarse_results)

    # Stage 3b: Fine refinement
    log("\n── Stage 3b: Fine Refinement ──")
    top_n = grid_config.get("fine_refinement_top_n", 20)
    for token, events in token_data.items():
        top_candidates = sorted(
            all_results[token],
            key=lambda r: r.survivor_score,
            reverse=True,
        )[:top_n]
        fine_results = fine_refinement(
            events, folds, token, model, top_candidates, of_config
        )
        all_results[token].extend(fine_results)

    # Stage 4: Darwinian
    log("\n── Stage 4: Darwinian Evolution ──")
    darwinian_config = {
        "darwinian_generations": grid_config.get("darwinian_generations", 3),
        "darwinian_population": grid_config.get("darwinian_population", 20),
        "perturbation_range": (0.05, 0.15),
        "darwinian_offspring": 3,
    }
    for token, events in token_data.items():
        evolved = darwinian_evolution(
            events,
            folds,
            token,
            model,
            all_results[token],
            of_config,
            darwinian_config,
        )
        all_results[token].extend(evolved)

    # Stage 4c: Experiments
    experiments = config.get("experiments", [])
    if experiments:
        log(f"\n── Stage 4c: Custom Experiments ({len(experiments)}) ──")
        for token, events in token_data.items():
            exp_results = run_experiments(
                events, folds, token, model, experiments, of_config
            )
            all_results[token].extend(exp_results)

    # Stage 5: Report
    log("\n── Stage 5: Morning Report ──")
    run_seconds = time.time() - start_time
    run_dir = generate_report(
        all_results,
        config,
        run_seconds,
        output_dir=config.get("output_dir"),
    )
    log(f"Report written to {run_dir}")
    log(f"Total runtime: {run_seconds / 60:.1f} minutes")

    survivors = [
        r
        for results in all_results.values()
        for r in results
        if not r.rejected
    ]
    best = max(survivors, key=lambda r: r.survivor_score) if survivors else None
    if best:
        log(
            f"Best survivor: {best.token} score={best.survivor_score:.3f} "
            f"(OOS={best.oos_score:+.3f}, consistency={best.oos_consistency:.0%})"
        )
    else:
        log("No survivors passed all gates")

    return {
        "run_dir": str(run_dir),
        "runtime_seconds": run_seconds,
        "results": all_results,
        "best": best,
    }