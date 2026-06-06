"""Token event data loading — synthetic, cached, or live ingestion."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from night_shift.core.logging import log
from night_shift.data.ingestion.bootstrap import bootstrap_token_history
from night_shift.data.ingestion.cache import read_cache, write_cache
from night_shift.data.ingestion.daily_aggregator import aggregate_transfers_to_daily, validate_daily_bars
from night_shift.data.ingestion.helius import fetch_token_daily_history, get_api_key
from night_shift.data.seed_manifest import resolve_token_list

# Re-export for backward compatibility
from night_shift.data.loader_synthetic import generate_synthetic_token_history  # noqa: F401


def _attach_metadata(df: pd.DataFrame, token_entry: Dict[str, Any], source: str) -> pd.DataFrame:
    out = df.copy()
    out.attrs["token"] = token_entry.get("symbol", token_entry["mint"])
    out.attrs["mint"] = token_entry["mint"]
    out.attrs["label"] = token_entry.get("label", "unknown")
    out.attrs["category"] = token_entry.get("category", "unknown")
    out.attrs["source"] = source
    out.attrs["synthetic"] = source in ("synthetic", "dry_run")
    return out


def _load_single_token(
    token_entry: Dict[str, Any],
    *,
    dry_run: bool = False,
    fetch_fresh: bool = False,
    history_days: int = 180,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    symbol = token_entry.get("symbol", token_entry["mint"])
    mint = token_entry["mint"]

    if dry_run or token_entry.get("label") == "synthetic":
        from night_shift.data.loader_synthetic import generate_synthetic_token_history

        log(f"  [{symbol}] Synthetic dry-run history ({history_days} days)")
        df = generate_synthetic_token_history(symbol, days=history_days)
        return _attach_metadata(df, token_entry, "dry_run")

    cached = None if fetch_fresh else read_cache(mint, cache_dir)
    if cached is not None and len(cached) >= history_days // 2:
        log(f"  [{symbol}] Loaded {len(cached)} days from cache")
        return _attach_metadata(validate_daily_bars(cached), token_entry, "cache")

    api_key = get_api_key()
    if fetch_fresh or api_key:
        if not api_key:
            log(f"  [{symbol}] Helius skipped (HELIUS_API_KEY not set)")
        else:
            log(f"  [{symbol}] Fetching from Helius...")
        events = fetch_token_daily_history(mint, days=history_days) if api_key else []
        if events:
            df = aggregate_transfers_to_daily(events, days=history_days)
            df = validate_daily_bars(df)
            if len(df) >= 14:
                write_cache(
                    mint,
                    df,
                    cache_dir,
                    metadata={
                        "symbol": symbol,
                        "label": token_entry.get("label"),
                        "source": "helius",
                    },
                )
                log(f"  [{symbol}] Cached {len(df)} days from Helius")
                return _attach_metadata(df, token_entry, "helius")

    log(f"  [{symbol}] Bootstrap from label={token_entry.get('label', 'unknown')}")
    df = bootstrap_token_history(token_entry, days=history_days)
    write_cache(mint, df, cache_dir, metadata={"symbol": symbol, "source": "bootstrap"})
    return _attach_metadata(validate_daily_bars(df), token_entry, "bootstrap")


def load_token_data(
    tokens: Optional[List[str]] = None,
    *,
    dry_run: bool = True,
    fetch_fresh: bool = False,
    use_seed_dataset: bool = False,
    seed_limit: Optional[int] = None,
    history_days: int = 180,
    cache_dir: Optional[str | Path] = None,
    manifest_path: Optional[str | Path] = None,
    data_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load token event histories for pipeline evaluation."""
    cfg = data_config or {}
    resolved = resolve_token_list(
        tokens=tokens,
        use_seed_dataset=use_seed_dataset,
        seed_limit=seed_limit or cfg.get("seed_limit"),
        manifest_path=manifest_path or cfg.get("seed_manifest"),
    )

    cache_path = Path(cache_dir) if cache_dir else None
    if cfg.get("cache_dir"):
        cache_path = Path(cfg["cache_dir"])

    data: Dict[str, pd.DataFrame] = {}
    for entry in resolved:
        symbol = entry.get("symbol", entry["mint"])
        data[symbol] = _load_single_token(
            entry,
            dry_run=dry_run,
            fetch_fresh=fetch_fresh or cfg.get("fetch_fresh", False),
            history_days=history_days or cfg.get("history_days", 180),
            cache_dir=cache_path,
        )
    return data


def prefetch_seed_dataset(
    limit: Optional[int] = None,
    fetch_fresh: bool = True,
    history_days: int = 180,
    manifest_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
) -> Dict[str, str]:
    """Prefetch and cache histories for all seed tokens. Returns source per token."""
    resolved = resolve_token_list(use_seed_dataset=True, seed_limit=limit, manifest_path=manifest_path)
    sources: Dict[str, str] = {}
    for entry in resolved:
        symbol = entry.get("symbol", entry["mint"])
        df = _load_single_token(
            entry,
            dry_run=False,
            fetch_fresh=fetch_fresh,
            history_days=history_days,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )
        sources[symbol] = df.attrs.get("source", "unknown")
    return sources