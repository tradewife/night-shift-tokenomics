"""Parameter grid generation and search stages."""

import hashlib
from itertools import product
from typing import Callable, Dict, List, Optional

import pandas as pd

from night_shift.core.logging import log
from night_shift.core.types import CandidateResult
from night_shift.models.base import DesignModel
from night_shift.validation.evaluate import evaluate_candidate
from night_shift.validation.folds import Fold


def grid_combos(grid: Dict[str, List]) -> List[Dict]:
    """Generate all combinations from a parameter grid."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def sample_grid_combos(
    grid: Dict[str, List],
    sample_size: int,
    *,
    seed: int = 42,
    token: str = "",
) -> List[Dict]:
    """Deterministic stratified sample of grid combos (stable across reruns)."""
    combos = grid_combos(grid)
    if sample_size <= 0 or sample_size >= len(combos):
        return combos

    digest = hashlib.sha256(f"{seed}:{token}".encode()).hexdigest()
    local_seed = int(digest[:8], 16)
    rng = __import__("random").Random(local_seed)
    indices = sorted(rng.sample(range(len(combos)), sample_size))
    return [combos[i] for i in indices]


def coarse_grid_search(
    events: pd.DataFrame,
    folds: List[Fold],
    token: str,
    model: DesignModel,
    of_config: Dict,
    coarse_window_days: int = 30,
    bars_per_day: int = 1,
    base_params: Optional[Dict] = None,
    coarse_sample_size: Optional[int] = None,
    coarse_sample_seed: int = 42,
) -> List[CandidateResult]:
    """Stage 1: fast single-window evaluation for rough ordering."""
    grid = model.param_grid(phase="coarse")
    if coarse_sample_size:
        combos = sample_grid_combos(
            grid, coarse_sample_size, seed=coarse_sample_seed, token=token
        )
        log(f"  Coarse grid: {len(combos)} sampled combos for {token}")
    else:
        combos = grid_combos(grid)
        log(f"  Coarse grid: {len(combos)} combos for {token}")

    window_bars = coarse_window_days * bars_per_day
    if len(events) > window_bars:
        coarse_fold = Fold(
            fold_num=0,
            train_start_idx=0,
            train_end_idx=max(0, len(events) - window_bars - 30),
            test_start_idx=len(events) - window_bars,
            test_end_idx=len(events),
            train_bars=max(0, len(events) - window_bars - 30),
            test_bars=window_bars,
        )
    else:
        coarse_fold = folds[-1]

    results: List[CandidateResult] = []
    for i, combo in enumerate(combos):
        params = {**(base_params or model.default_params()), **combo}
        cr = evaluate_candidate(
            events,
            [coarse_fold],
            params,
            token,
            model,
            of_config,
            compute_fragility=False,
            skip_is=True,
        )
        results.append(cr)

        if (i + 1) % 500 == 0:
            best = max(r.survivor_score for r in results)
            log(f"    [{token}] {i + 1}/{len(combos)} evaluated... best survivor={best:.3f}")

    passed = sum(1 for r in results if not r.rejected)
    log(f"    [{token}] Coarse done. {passed}/{len(results)} passed filters")
    return results


def fine_refinement(
    events: pd.DataFrame,
    folds: List[Fold],
    token: str,
    model: DesignModel,
    top_candidates: List[CandidateResult],
    of_config: Dict,
) -> List[CandidateResult]:
    """Stage 2: full walk-forward on top candidates with fine param sweep."""
    results: List[CandidateResult] = []
    seen_keys: set = set()
    fine_grid = model.param_grid(phase="fine")

    for parent in top_candidates:
        if parent.rejected:
            continue
        base = dict(parent.params)
        for key in fine_grid:
            base.pop(key, None)

        for combo in grid_combos(fine_grid):
            params = {**base, **combo}
            key = tuple(sorted(params.items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cr = evaluate_candidate(
                events,
                folds,
                params,
                token,
                model,
                of_config,
                compute_fragility=True,
            )
            results.append(cr)

    log(f"    [{token}] Fine refinement: {len(results)} candidates on {len(folds)} folds")
    return results


def run_experiments(
    events: pd.DataFrame,
    folds: List[Fold],
    token: str,
    model: DesignModel,
    experiments: List[Dict],
    of_config: Dict,
) -> List[CandidateResult]:
    """Run declarative param-override experiments from config."""
    results: List[CandidateResult] = []
    for exp in experiments:
        name = exp.get("name", "unnamed")
        param_grid = exp.get("params", {})
        if not param_grid:
            continue
        log(f"    [{token}] Experiment: {name} ({len(grid_combos(param_grid))} combos)")
        for combo in grid_combos(param_grid):
            params = {**model.default_params(), **combo}
            cr = evaluate_candidate(
                events,
                folds,
                params,
                token,
                model,
                of_config,
                compute_fragility=True,
            )
            results.append(cr)
    return results