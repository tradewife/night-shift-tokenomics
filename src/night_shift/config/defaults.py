"""Default configuration values for the tokenomics research pipeline."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "night_config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "night_results"

WFA_DEFAULTS = {
    "num_folds": 6,
    "test_fold_days": 30,
    "min_events_per_fold": 5,
    "warmup_bars": 30,
    "bars_per_day": 1,
}

OVERFITTING_DEFAULTS = {
    "max_is_oos_gap": 0.5,
    "min_oos_consistency": 0.50,
    "max_fragility": 0.4,
}

DARWINIAN_DEFAULTS = {
    "generations": 3,
    "population": 20,
    "perturbation_range": (0.05, 0.15),
    "offspring_per_parent": 3,
}

SCORE_CAP = 100.0