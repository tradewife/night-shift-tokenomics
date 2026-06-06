"""Load and merge night shift configuration from JSON."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from night_shift.config.defaults import (
    DARWINIAN_DEFAULTS,
    DEFAULT_CONFIG_PATH,
    OVERFITTING_DEFAULTS,
    WFA_DEFAULTS,
)


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load night_config.json and merge with defaults."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config: Dict[str, Any] = {}
    if path.exists():
        with path.open() as f:
            config = json.load(f)

    config.setdefault("wfa", {}).update(
        {k: v for k, v in WFA_DEFAULTS.items() if k not in config.get("wfa", {})}
    )
    config.setdefault("overfitting", {}).update(
        {k: v for k, v in OVERFITTING_DEFAULTS.items() if k not in config.get("overfitting", {})}
    )
    grid = config.setdefault("grid_search", {})
    for key, value in DARWINIAN_DEFAULTS.items():
        grid.setdefault(key if key != "offspring_per_parent" else "darwinian_offspring", value)
    if "darwinian_generations" not in grid:
        grid["darwinian_generations"] = DARWINIAN_DEFAULTS["generations"]
    if "darwinian_population" not in grid:
        grid["darwinian_population"] = DARWINIAN_DEFAULTS["population"]

    return config