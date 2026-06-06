"""Expanding-window walk-forward fold creation."""

from dataclasses import dataclass
from typing import List


@dataclass
class Fold:
    """One train/test temporal split."""

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
    test_fold_days: int,
    warmup_bars: int = 30,
    bars_per_day: int = 1,
) -> List[Fold]:
    """
    Create non-overlapping expanding-window walk-forward folds.

    Fold 1: [TRAIN====][TEST]
    Fold 2: [TRAIN============][TEST]
    ...
    Every bar appears in exactly one test fold.
    """
    test_bars = test_fold_days * bars_per_day

    if total_bars <= warmup_bars + test_bars:
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
    max_folds = usable_bars // test_bars
    actual_folds = min(num_folds, max_folds)

    if actual_folds < max_folds:
        bars_per_fold = test_bars
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