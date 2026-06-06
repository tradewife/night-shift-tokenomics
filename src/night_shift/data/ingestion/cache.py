"""Parquet cache for token daily history."""

from pathlib import Path
from typing import Optional

import pandas as pd

from night_shift.config.defaults import PROJECT_ROOT

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "tokens" / "cache"


def cache_path(mint: str, cache_dir: Optional[Path] = None) -> Path:
    base = cache_dir or DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    safe_mint = mint.replace("/", "_")
    return base / f"{safe_mint}.parquet"


def read_cache(mint: str, cache_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    path = cache_path(mint, cache_dir)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def write_cache(
    mint: str,
    df: pd.DataFrame,
    cache_dir: Optional[Path] = None,
    metadata: Optional[dict] = None,
) -> Path:
    path = cache_path(mint, cache_dir)
    out = df.copy()
    if metadata:
        for key, value in metadata.items():
            out.attrs[key] = value
    out.to_parquet(path, index=False)
    return path