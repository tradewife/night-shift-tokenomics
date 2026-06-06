"""Smoke tests for the research pipeline."""

from night_shift.search.grid import grid_combos
from night_shift.taxonomy.parameter_space import COARSE_GRID
from night_shift.validation.folds import create_folds


def test_create_folds():
    folds = create_folds(total_bars=180, num_folds=6, test_fold_days=30, warmup_bars=30)
    assert len(folds) >= 1
    assert folds[0].train_start_idx == 0


def test_coarse_grid_size():
    combos = grid_combos(COARSE_GRID)
    assert len(combos) > 100


def test_dry_run_pipeline():
    from night_shift.core.pipeline import run_night_shift

    result = run_night_shift(tokens=["TEST-TOKEN"], dry_run=True)
    assert result["runtime_seconds"] > 0
    assert result["best"] is not None or result["run_dir"]