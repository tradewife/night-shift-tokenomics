"""Load and query the curated seed token manifest."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from night_shift.config.defaults import PROJECT_ROOT

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "seed_tokens.json"


def load_seed_manifest(path: Optional[str | Path] = None) -> Dict[str, Any]:
    manifest_path = Path(path) if path else DEFAULT_MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(f"Seed manifest not found: {manifest_path}")
    with manifest_path.open() as f:
        return json.load(f)


def list_seed_tokens(
    path: Optional[str | Path] = None,
    label: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    manifest = load_seed_manifest(path)
    tokens = manifest.get("tokens", [])
    if label:
        tokens = [t for t in tokens if t.get("label") == label]
    if limit:
        tokens = tokens[:limit]
    return tokens


def get_token_entry(symbol_or_mint: str, path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    manifest = load_seed_manifest(path)
    for token in manifest.get("tokens", []):
        if token.get("symbol") == symbol_or_mint or token.get("mint") == symbol_or_mint:
            return token
    return None


def resolve_token_list(
    tokens: Optional[List[str]] = None,
    use_seed_dataset: bool = False,
    seed_limit: Optional[int] = None,
    manifest_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Resolve CLI/config token identifiers to manifest entries."""
    if use_seed_dataset:
        return list_seed_tokens(manifest_path, limit=seed_limit)

    if not tokens:
        return list_seed_tokens(manifest_path, limit=seed_limit or 5)

    resolved: List[Dict[str, Any]] = []
    for token_id in tokens:
        if token_id.startswith("SYNTHETIC-"):
            resolved.append(
                {
                    "symbol": token_id,
                    "mint": token_id,
                    "label": "synthetic",
                    "category": "synthetic",
                }
            )
            continue
        entry = get_token_entry(token_id, manifest_path)
        if entry:
            resolved.append(entry)
        else:
            resolved.append(
                {
                    "symbol": token_id,
                    "mint": token_id,
                    "label": "unknown",
                    "category": "unknown",
                }
            )
    return resolved