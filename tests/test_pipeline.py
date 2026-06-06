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

    result = run_night_shift(tokens=["SYNTHETIC-ALPHA"], dry_run=True)
    assert result["runtime_seconds"] > 0
    assert result["best"] is not None or result["run_dir"]


def test_live_data_and_evaluator():
    from night_shift.data.loader import load_token_data
    from night_shift.models.registry import get_model
    from night_shift.validation.evaluate import evaluate_candidate
    from night_shift.validation.folds import create_folds

    data = load_token_data(["JUP"], dry_run=False)
    assert "JUP" in data
    assert data["JUP"].attrs.get("source") in ("bootstrap", "cache", "helius")

    model = get_model(use_stub=False)
    folds = create_folds(len(data["JUP"]), num_folds=3, test_fold_days=30, warmup_bars=30)
    result = evaluate_candidate(
        data["JUP"],
        folds,
        model.default_params(),
        "JUP",
        model,
        {"max_is_oos_gap": 0.5, "min_oos_consistency": 0.5, "min_events_per_fold": 1},
        compute_fragility=False,
    )
    assert result.token == "JUP"
    assert result.oos_avg_events_per_fold >= 0