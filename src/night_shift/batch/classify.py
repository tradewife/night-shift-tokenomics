"""Phase B: resume-safe per-token classification for the seed dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from night_shift.config.loader import load_config
from night_shift.core.logging import log
from night_shift.core.pipeline import run_night_shift
from night_shift.core.types import CandidateResult
from night_shift.data.backfill_manifest import load_manifest
from night_shift.data.seed_manifest import list_seed_tokens
from night_shift.export.observatory import (
    build_rejection_rows,
    build_token_export_record,
    export_observatory_artifacts,
    write_json,
)


def _export_dir(base: str | Path, run_id: Optional[str] = None) -> Path:
    base_path = Path(base)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return base_path / f"v1_{run_id}"


def _batch_manifest_path(export_dir: Path) -> Path:
    return export_dir / "batch_manifest.json"


def load_batch_manifest(export_dir: Path) -> Dict[str, Any]:
    path = _batch_manifest_path(export_dir)
    if not path.exists():
        return {"schema_version": "1.0", "tokens": {}}
    with path.open() as f:
        return json.load(f)


def save_batch_manifest(export_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(_batch_manifest_path(export_dir), manifest)


def _serialize_pipeline_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe subset for per-token detail files."""
    out: Dict[str, Any] = {
        "run_dir": result.get("run_dir"),
        "runtime_seconds": result.get("runtime_seconds"),
        "resilience_rankings": {},
        "robustness_results": result.get("robustness_results"),
        "agent_stress_results": {
            k: v for k, v in (result.get("agent_stress_results") or {}).items() if k != "_by_params"
        },
        "results": {},
    }
    for token, candidates in result.get("results", {}).items():
        out["results"][token] = [
            {
                "survivor_score": c.survivor_score,
                "oos_score": c.oos_score,
                "oos_consistency": c.oos_consistency,
                "rejected": c.rejected,
                "rejection_reason": c.rejection_reason,
                "regime_summary": c.regime_summary,
                "params": c.params,
                "folds": c.folds,
            }
            for c in sorted(candidates, key=lambda x: x.survivor_score, reverse=True)[:30]
        ]
    for token, rankings in result.get("resilience_rankings", {}).items():
        out["resilience_rankings"][token] = [
            {
                "resilience_score": r["resilience_score"],
                "survivor_score": r["survivor_score"],
                "components": {
                    "value_accrual": r["components"].value_accrual,
                    "attack_resistance": r["components"].attack_resistance,
                    "stress_sustainability": r["components"].stress_sustainability,
                    "total": r["components"].total,
                },
                "params": r["params"],
            }
            for r in rankings
        ]
    return out


def run_classify_phase(
    config_path: Optional[str] = None,
    *,
    export_base: str | Path = "data/exports",
    run_id: Optional[str] = None,
    resume: bool = True,
    seed_manifest_path: Optional[str | Path] = None,
    backfill_manifest_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Classify all seed tokens sequentially with resume; export Observatory artifacts."""
    config = load_config(config_path)
    export_dir = _export_dir(export_base, run_id)
    tokens_dir = export_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    batch_manifest = load_batch_manifest(export_dir) if resume else {"schema_version": "1.0", "tokens": {}}
    backfill_manifest = load_manifest(backfill_manifest_path)
    seed_tokens = list_seed_tokens(seed_manifest_path)

    log(f"Classify phase: {len(seed_tokens)} tokens → {export_dir}")

    token_records: List[Dict[str, Any]] = []
    rejection_rows: List[Dict[str, Any]] = []

    for entry in seed_tokens:
        symbol = entry.get("symbol", entry["mint"])
        prior = batch_manifest.get("tokens", {}).get(symbol, {})
        if resume and prior.get("status") == "done":
            token_path = tokens_dir / f"{symbol}.json"
            if token_path.exists():
                with token_path.open() as f:
                    saved = json.load(f)
                token_records.append(saved["dataset_record"])
                rejection_rows.extend(saved.get("rejection_rows", []))
                log(f"  [{symbol}] Skipped (already classified)")
                continue

        batch_manifest.setdefault("tokens", {})[symbol] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        save_batch_manifest(export_dir, batch_manifest)

        try:
            log(f"  [{symbol}] Running pipeline...")
            result = run_night_shift(
                config_path=config_path,
                tokens=[symbol],
                dry_run=False,
                fetch_fresh=False,
            )
            backfill_record = backfill_manifest.get("tokens", {}).get(symbol, {})
            dataset_record = build_token_export_record(
                symbol, entry, result, backfill_record=backfill_record
            )
            results = result.get("results", {}).get(symbol, [])
            token_rejections = build_rejection_rows(symbol, results)

            token_payload = {
                "symbol": symbol,
                "dataset_record": dataset_record,
                "rejection_rows": token_rejections,
                "pipeline": _serialize_pipeline_result(result),
            }
            write_json(tokens_dir / f"{symbol}.json", token_payload)

            token_records.append(dataset_record)
            rejection_rows.extend(token_rejections)

            batch_manifest["tokens"][symbol] = {
                "status": "done",
                "started_at": batch_manifest["tokens"][symbol].get("started_at"),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "engine_verdict": dataset_record["engine_verdict"],
                "taxonomy_signal": dataset_record["taxonomy_signal"],
            }
            save_batch_manifest(export_dir, batch_manifest)
            log(
                f"  [{symbol}] Done — {dataset_record['engine_verdict']}, "
                f"signal={dataset_record['taxonomy_signal']}, "
                f"resilience={dataset_record['best_resilience_score']:.1f}"
            )
        except Exception as exc:
            batch_manifest["tokens"][symbol] = {
                "status": "failed",
                "error": str(exc),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            save_batch_manifest(export_dir, batch_manifest)
            log(f"  [{symbol}] FAILED: {exc}")

    # Always export aggregate artifacts (even if all rejected)
    export_observatory_artifacts(
        export_dir,
        token_records,
        config=config,
        rejection_rows=rejection_rows,
        backfill_manifest=backfill_manifest,
        batch_manifest=batch_manifest,
    )
    log(f"Observatory export written to {export_dir}")
    return {
        "export_dir": str(export_dir),
        "token_count": len(token_records),
        "survivors": sum(1 for r in token_records if r.get("engine_verdict") == "survivor"),
    }