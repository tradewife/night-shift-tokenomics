"""Preflight regime diversity audit on cached daily bars."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from night_shift.validation.folds import Fold, create_folds
from night_shift.validation.regimes import REGIME_UNKNOWN, tag_window_regime


def preflight_regime_distribution(
    events: pd.DataFrame,
    *,
    num_folds: int = 6,
    test_fold_days: int = 30,
    warmup_bars: int = 30,
) -> Dict[str, Any]:
    """Tag WFA test windows on historical data and count regime diversity."""
    if events is None or len(events) < warmup_bars + test_fold_days:
        return {
            "historical_regime_counts": {},
            "distinct_regimes": 0,
            "low_regime_diversity": True,
            "fold_regimes": [],
        }

    folds = create_folds(
        total_bars=len(events),
        num_folds=num_folds,
        test_fold_days=test_fold_days,
        warmup_bars=warmup_bars,
    )
    fold_regimes: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for fold in folds:
        regime, features = tag_window_regime(
            events, fold.test_start_idx, fold.test_end_idx
        )
        fold_regimes.append(
            {
                "fold": fold.fold_num,
                "regime": regime,
                "test_start": fold.test_start_idx,
                "test_end": fold.test_end_idx,
                "features": features,
            }
        )
        if regime != REGIME_UNKNOWN:
            counts[regime] += 1

    distinct = len(counts)
    return {
        "historical_regime_counts": dict(sorted(counts.items())),
        "distinct_regimes": distinct,
        "low_regime_diversity": distinct < 3,
        "fold_regimes": fold_regimes,
    }