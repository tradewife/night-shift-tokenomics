"""Resume-safe manifest for Helius backfill across the seed dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from night_shift.config.defaults import PROJECT_ROOT

DEFAULT_BACKFILL_MANIFEST = PROJECT_ROOT / "data" / "tokens" / "backfill_manifest.json"
TERMINAL_STATUSES = frozenset({"helius", "cache", "bootstrap"})


def manifest_path(path: Optional[str | Path] = None) -> Path:
    return Path(path) if path else DEFAULT_BACKFILL_MANIFEST


def load_manifest(path: Optional[str | Path] = None) -> Dict[str, Any]:
    p = manifest_path(path)
    if not p.exists():
        return {
            "schema_version": "1.0",
            "history_days": 270,
            "updated_at": None,
            "tokens": {},
        }
    with p.open() as f:
        return json.load(f)


def save_manifest(data: Dict[str, Any], path: Optional[str | Path] = None) -> Path:
    p = manifest_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(p)
    return p


def get_token_record(manifest: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    return manifest.get("tokens", {}).get(symbol)


def should_skip_token(
    manifest: Dict[str, Any],
    symbol: str,
    *,
    force: bool = False,
    min_bars: int = 150,
) -> bool:
    if force:
        return False
    record = get_token_record(manifest, symbol)
    if not record:
        return False
    if record.get("status") not in TERMINAL_STATUSES:
        return False
    return int(record.get("bars", 0)) >= min_bars


def update_token_record(
    manifest: Dict[str, Any],
    symbol: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    manifest.setdefault("tokens", {})[symbol] = record
    return manifest