"""Sidecar metadata for cached token daily bars."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from night_shift.data.ingestion.cache import cache_path


def meta_path(mint: str, cache_dir: Optional[Path] = None) -> Path:
    return cache_path(mint, cache_dir).with_suffix(".meta.json")


def read_meta(mint: str, cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = meta_path(mint, cache_dir)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def write_meta(
    mint: str,
    payload: Dict[str, Any],
    cache_dir: Optional[Path] = None,
) -> Path:
    path = meta_path(mint, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".meta.json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)
    return path


def build_meta_payload(
    *,
    symbol: str,
    mint: str,
    source: str,
    bars: int,
    label: str = "unknown",
    category: str = "unknown",
    history_days: int = 180,
    regime_preflight: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    attempts: int = 1,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "symbol": symbol,
        "mint": mint,
        "label": label,
        "category": category,
        "source": source,
        "bars": bars,
        "history_days": history_days,
        "synthetic_fallback": source == "bootstrap",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "error": error,
        "regime_preflight": regime_preflight or {},
    }