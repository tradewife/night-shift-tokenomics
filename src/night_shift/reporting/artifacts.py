"""Output artifact path management."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from night_shift.config.defaults import DEFAULT_OUTPUT_DIR


def get_run_dir(output_dir: Optional[str | Path] = None) -> Path:
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_dir = base / date_str
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir