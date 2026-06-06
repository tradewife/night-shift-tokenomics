"""Darwinian evolutionary refinement of design candidates."""

import random
from typing import Dict, List

import pandas as pd

from night_shift.core.logging import log
from night_shift.core.types import CandidateResult
from night_shift.models.base import DesignModel
from night_shift.validation.evaluate import evaluate_candidate
from night_shift.validation.folds import Fold


def darwinian_evolution(
    events: pd.DataFrame,
    folds: List[Fold],
    token: str,
    model: DesignModel,
    population: List[CandidateResult],
    of_config: Dict,
    config: Dict,
) -> List[CandidateResult]:
    """Stage 3: perturb top survivors and select by fitness across generations."""
    generations = config.get("darwinian_generations", config.get("generations", 3))
    pop_size = config.get("darwinian_population", config.get("population", 20))
    perturb_range = config.get("perturbation_range", (0.05, 0.15))
    offspring_per_parent = config.get("darwinian_offspring", config.get("offspring_per_parent", 3))

    current_gen = sorted(
        [r for r in population if not r.rejected],
        key=lambda r: r.survivor_score,
        reverse=True,
    )[:pop_size]

    if not current_gen:
        log(f"    [{token}] No survivors for Darwinian evolution")
        return []

    all_survivors = list(current_gen)

    for gen in range(generations):
        offspring: List[CandidateResult] = []
        for parent in current_gen:
            for _ in range(offspring_per_parent):
                params = dict(parent.params)
                numeric_keys = [
                    k for k, v in params.items()
                    if isinstance(v, (int, float)) and not model.is_discrete(k)
                ]
                if not numeric_keys:
                    continue
                key = random.choice(numeric_keys)
                delta = random.uniform(*perturb_range) * random.choice([-1, 1])
                original = params[key]
                if isinstance(original, int):
                    params[key] = max(1, int(original * (1 + delta)))
                else:
                    floor = model.param_floor(key)
                    params[key] = max(floor, round(original * (1 + delta), 4))

                cr = evaluate_candidate(
                    events,
                    folds,
                    params,
                    token,
                    model,
                    of_config,
                    compute_fragility=True,
                )
                offspring.append(cr)

        combined = current_gen + offspring
        combined.sort(key=lambda r: r.survivor_score, reverse=True)
        current_gen = combined[:pop_size]
        all_survivors.extend(current_gen)

        log(
            f"    [{token}] Darwinian gen {gen + 1}/{generations}: "
            f"{len(offspring)} offspring, best survivor={current_gen[0].survivor_score:.3f}"
        )

    seen: set = set()
    unique: List[CandidateResult] = []
    for result in sorted(all_survivors, key=lambda r: r.survivor_score, reverse=True):
        key = tuple(sorted(result.params.items()))
        if key not in seen:
            seen.add(key)
            unique.append(result)

    return unique[: pop_size * 2]