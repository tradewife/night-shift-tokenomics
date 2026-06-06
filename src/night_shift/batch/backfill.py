"""Phase A: resume-safe Helius backfill for the full seed manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from night_shift.core.logging import log
from night_shift.data.backfill_manifest import (
    load_manifest,
    save_manifest,
    should_skip_token,
    update_token_record,
)
from night_shift.data.ingestion.cache import read_cache
from night_shift.data.ingestion.helius import get_api_key
from night_shift.data.ingestion.meta import build_meta_payload, write_meta
from night_shift.data.loader import _load_single_token
from night_shift.data.regime_preflight import preflight_regime_distribution
from night_shift.data.seed_manifest import list_seed_tokens


def run_backfill_phase(
    *,
    history_days: int = 270,
    cache_dir: Optional[str | Path] = None,
    manifest_path: Optional[str | Path] = None,
    seed_manifest_path: Optional[str | Path] = None,
    force: bool = False,
    resume: bool = True,
    min_bars: int = 150,
    wfa_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Backfill all seed tokens with resume-safe manifest tracking.

    Returns summary dict with per-token sources.
    """
    cache_path = Path(cache_dir) if cache_dir else None
    wfa = wfa_config or {}
    manifest = load_manifest(manifest_path) if resume else {
        "schema_version": "1.0",
        "history_days": history_days,
        "tokens": {},
    }
    manifest["history_days"] = history_days

    tokens = list_seed_tokens(seed_manifest_path)
    log(f"Backfill phase: {len(tokens)} seed tokens, history_days={history_days}")

    sources: Dict[str, str] = {}
    for entry in tokens:
        symbol = entry.get("symbol", entry["mint"])
        mint = entry["mint"]

        if should_skip_token(manifest, symbol, force=force, min_bars=min_bars):
            record = manifest["tokens"][symbol]
            sources[symbol] = record.get("status", record.get("source", "cache"))
            log(f"  [{symbol}] Skipped (cached, {record.get('bars', '?')} bars)")
            continue

        prior = manifest.get("tokens", {}).get(symbol, {})
        attempts = int(prior.get("attempts", 0)) + 1
        log(f"  [{symbol}] Backfill attempt {attempts}...")

        try:
            cached_bars = len(read_cache(mint, cache_path) or [])
            target_bars = max(min_bars, int(history_days * 0.85))
            needs_fetch = force or cached_bars < target_bars
            df = _load_single_token(
                entry,
                dry_run=False,
                fetch_fresh=needs_fetch,
                history_days=history_days,
                cache_dir=cache_path,
            )
            if needs_fetch and df.attrs.get("source") == "bootstrap" and get_api_key():
                log(f"  [{symbol}] Helius returned no events; using bootstrap fallback")
            source = df.attrs.get("source", "unknown")
            bars = len(df)
            preflight = preflight_regime_distribution(
                df,
                num_folds=wfa.get("num_folds", 6),
                test_fold_days=wfa.get("test_fold_days", 30),
                warmup_bars=wfa.get("warmup_bars", 30),
            )

            status = source if source in ("helius", "bootstrap") else "cache"
            if bars < min_bars:
                status = "bootstrap" if source == "bootstrap" else status

            record = {
                "mint": mint,
                "status": status,
                "source": source,
                "bars": bars,
                "label": entry.get("label", "unknown"),
                "category": entry.get("category", "unknown"),
                "distinct_regimes": preflight.get("distinct_regimes", 0),
                "low_regime_diversity": preflight.get("low_regime_diversity", True),
                "historical_regime_counts": preflight.get("historical_regime_counts", {}),
                "synthetic_fallback": source == "bootstrap",
                "attempts": attempts,
                "error": None,
            }
            update_token_record(manifest, symbol, record)
            save_manifest(manifest, manifest_path)

            write_meta(
                mint,
                build_meta_payload(
                    symbol=symbol,
                    mint=mint,
                    source=source,
                    bars=bars,
                    label=entry.get("label", "unknown"),
                    category=entry.get("category", "unknown"),
                    history_days=history_days,
                    regime_preflight=preflight,
                    attempts=attempts,
                ),
                cache_path,
            )
            sources[symbol] = source
            log(
                f"  [{symbol}] {source} — {bars} bars, "
                f"regimes={preflight.get('distinct_regimes', 0)} "
                f"{preflight.get('historical_regime_counts', {})}"
            )
        except Exception as exc:
            record = {
                "mint": mint,
                "status": "failed",
                "source": "failed",
                "bars": len(read_cache(mint, cache_path) or []),
                "attempts": attempts,
                "error": str(exc),
            }
            update_token_record(manifest, symbol, record)
            save_manifest(manifest, manifest_path)
            sources[symbol] = "failed"
            log(f"  [{symbol}] FAILED: {exc}")

    terminal = sum(1 for s in sources.values() if s in ("helius", "cache", "bootstrap"))
    failed = sum(1 for s in sources.values() if s == "failed")
    log(f"Backfill complete: {terminal} ok, {failed} failed, {len(sources)} total")
    return {"sources": sources, "manifest": manifest}