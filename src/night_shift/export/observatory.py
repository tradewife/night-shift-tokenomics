"""Public Observatory export artifacts from batch classification results."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from night_shift.config.defaults import PROJECT_ROOT
from night_shift.scoring.resilience import ResilienceComponents, compute_resilience_score


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _candidate_to_dict(candidate: Any) -> Dict[str, Any]:
    return {
        "token": candidate.token,
        "survivor_score": candidate.survivor_score,
        "oos_score": candidate.oos_score,
        "oos_consistency": candidate.oos_consistency,
        "overfitting_score": candidate.overfitting_score,
        "fragility": candidate.fragility,
        "rejected": candidate.rejected,
        "rejection_reason": candidate.rejection_reason,
        "regime_summary": candidate.regime_summary,
        "params": candidate.params,
        "folds": candidate.folds,
    }


def _primary_rejection_gate(rejection_reason: str) -> str:
    if not rejection_reason:
        return "none"
    if "regime_gate" in rejection_reason:
        return "regime_gate"
    if "overfitting_score" in rejection_reason or "oos_consistency" in rejection_reason:
        return "overfitting"
    if "aggregate_attack_surface" in rejection_reason or "death_spiral" in rejection_reason:
        return "agent_stress"
    if "PBO" in rejection_reason or "monte_carlo" in rejection_reason or "sensitivity" in rejection_reason:
        return "robustness"
    return "other"


def _taxonomy_signal(
    *,
    engine_verdict: str,
    resilience_score: float,
    rejection_reason: str,
    synthetic_fallback: bool,
    history_days: int,
) -> str:
    if synthetic_fallback or history_days < 150:
        return "insufficient_data"
    if engine_verdict == "survivor":
        if resilience_score >= 60:
            return "resilient_pattern"
        if resilience_score >= 40:
            return "mixed"
        return "fragile"
    gate = _primary_rejection_gate(rejection_reason)
    if gate == "regime_gate":
        return "regime_fragile"
    if gate == "overfitting":
        return "narrative_overfit"
    if gate in ("robustness", "agent_stress"):
        return "stress_fragile"
    return "fragile"


def _seed_agreement(seed_label: str, taxonomy_signal: str) -> str:
    resilient_signals = {"resilient_pattern", "mixed"}
    fragile_signals = {"fragile", "regime_fragile", "stress_fragile", "narrative_overfit", "insufficient_data"}
    if seed_label == "resilient" and taxonomy_signal in resilient_signals:
        return "agrees"
    if seed_label == "fragile" and taxonomy_signal in fragile_signals:
        return "agrees"
    return "disagrees"


def _confidence(data_source: str, distinct_regimes: int, synthetic_fallback: bool) -> str:
    if synthetic_fallback or data_source == "bootstrap":
        return "low"
    if distinct_regimes >= 3 and data_source == "helius":
        return "high"
    if distinct_regimes >= 2:
        return "medium"
    return "low"


def build_token_export_record(
    symbol: str,
    token_entry: Dict[str, Any],
    pipeline_result: Dict[str, Any],
    backfill_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one dataset.jsonl row from a per-token pipeline run."""
    all_results = pipeline_result.get("results", {})
    results = all_results.get(symbol, [])
    survivors = [r for r in results if not r.rejected]
    best = max(results, key=lambda r: r.survivor_score) if results else None
    best_survivor = max(survivors, key=lambda r: r.survivor_score) if survivors else best

    rankings = pipeline_result.get("resilience_rankings", {}).get(symbol, [])
    resilience_score = 0.0
    components: Optional[ResilienceComponents] = None
    if rankings:
        resilience_score = rankings[0]["resilience_score"]
        components = rankings[0]["components"]
    elif best_survivor:
        components = compute_resilience_score(best_survivor)
        resilience_score = components.total

    backfill = backfill_record or {}
    data_source = backfill.get("source", "unknown")
    history_days = int(backfill.get("bars", 0))
    synthetic_fallback = bool(backfill.get("synthetic_fallback", data_source == "bootstrap"))
    historical_counts = backfill.get("historical_regime_counts", {})
    distinct_regimes = int(backfill.get("distinct_regimes", 0))

    engine_verdict = "survivor" if survivors else "rejected"
    rejection_reason = best_survivor.rejection_reason if best_survivor else ""
    taxonomy_signal = _taxonomy_signal(
        engine_verdict=engine_verdict,
        resilience_score=resilience_score,
        rejection_reason=rejection_reason,
        synthetic_fallback=synthetic_fallback,
        history_days=history_days,
    )

    regime_summary = best_survivor.regime_summary if best_survivor else {}
    components_dict = {}
    if components:
        components_dict = {
            "value_accrual": components.value_accrual,
            "holder_retention": components.holder_retention,
            "attack_resistance_base": components.attack_resistance_base,
            "security_penalty": components.security_penalty,
            "attack_resistance": components.attack_resistance,
            "stress_sustainability": components.stress_sustainability,
            "agent_attack_surface": components.agent_attack_surface,
            "transparency": components.transparency,
            "extraction_penalty": components.extraction_penalty,
        }

    return {
        "symbol": symbol,
        "mint": token_entry.get("mint"),
        "seed_label": token_entry.get("label", "unknown"),
        "seed_category": token_entry.get("category", "unknown"),
        "data_source": data_source,
        "synthetic_fallback": synthetic_fallback,
        "history_days": history_days,
        "historical_regime_counts": historical_counts,
        "distinct_regimes": distinct_regimes,
        "low_regime_diversity": bool(backfill.get("low_regime_diversity", distinct_regimes < 3)),
        "candidates_evaluated": len(results),
        "survivors": len(survivors),
        "best_survivor_score": best.survivor_score if best else 0.0,
        "best_resilience_score": resilience_score,
        "engine_verdict": engine_verdict,
        "primary_rejection_gate": _primary_rejection_gate(rejection_reason),
        "rejection_reason": rejection_reason,
        "regime_summary": regime_summary,
        "best_params": best_survivor.params if best_survivor else {},
        "resilience_components": components_dict,
        "taxonomy_signal": taxonomy_signal,
        "seed_agreement": _seed_agreement(token_entry.get("label", "unknown"), taxonomy_signal),
        "confidence": _confidence(data_source, distinct_regimes, synthetic_fallback),
    }


def build_rejection_rows(symbol: str, results: List[Any], limit: int = 20) -> List[Dict[str, Any]]:
    survivors = [r for r in results if not r.rejected]
    rejected = sorted(
        [r for r in results if r.rejected],
        key=lambda r: r.survivor_score,
        reverse=True,
    )[:limit]
    rows = [_candidate_to_dict(r) for r in survivors]
    rows.extend(_candidate_to_dict(r) for r in rejected)
    for row in rows:
        row["symbol"] = symbol
    return rows


def build_regime_counts(token_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    historical = Counter()
    engine_profitable = Counter()
    per_token: Dict[str, Any] = {}

    for rec in token_records:
        symbol = rec["symbol"]
        for regime, count in rec.get("historical_regime_counts", {}).items():
            historical[regime] += count
        profitable = rec.get("regime_summary", {}).get("profitable_regimes", [])
        for regime in profitable:
            engine_profitable[regime] += 1
        per_token[symbol] = {
            "historical": rec.get("historical_regime_counts", {}),
            "engine_profitable": profitable,
            "distinct_regimes": rec.get("distinct_regimes", 0),
            "gate_passed": rec.get("regime_summary", {}).get("gate_passed"),
        }

    passing = sum(
        1 for rec in token_records if rec.get("regime_summary", {}).get("gate_passed")
    )
    failing = sum(
        1
        for rec in token_records
        if rec.get("engine_verdict") == "rejected"
        and rec.get("primary_rejection_gate") == "regime_gate"
    )

    return {
        "aggregate": {
            "historical_regimes_across_tokens": dict(sorted(historical.items())),
            "profitable_regimes_in_engine": dict(sorted(engine_profitable.items())),
            "tokens_failing_regime_gate": failing,
            "tokens_passing_regime_gate": passing,
            "token_count": len(token_records),
        },
        "per_token": per_token,
    }


def build_taxonomy_labels(token_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = {}
    signal_counts = Counter()
    agreement_counts = Counter()
    for rec in token_records:
        symbol = rec["symbol"]
        signal = rec.get("taxonomy_signal", "fragile")
        signal_counts[signal] += 1
        agreement = rec.get("seed_agreement", "disagrees")
        agreement_counts[agreement] += 1
        labels[symbol] = {
            "symbol": symbol,
            "seed_label": rec.get("seed_label"),
            "seed_category": rec.get("seed_category"),
            "taxonomy_signal": signal,
            "seed_agreement": agreement,
            "confidence": rec.get("confidence", "low"),
            "resilience_score": rec.get("best_resilience_score", 0.0),
            "engine_verdict": rec.get("engine_verdict"),
            "primary_rejection_gate": rec.get("primary_rejection_gate"),
            "data_source": rec.get("data_source"),
            "synthetic_fallback": rec.get("synthetic_fallback", False),
        }
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_commit": _git_commit(),
        "signal_counts": dict(sorted(signal_counts.items())),
        "seed_agreement_counts": dict(sorted(agreement_counts.items())),
        "tokens": labels,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def export_observatory_artifacts(
    export_dir: str | Path,
    token_records: List[Dict[str, Any]],
    *,
    config: Dict[str, Any],
    rejection_rows: List[Dict[str, Any]],
    backfill_manifest: Optional[Dict[str, Any]] = None,
    batch_manifest: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write all public Observatory artifacts unconditionally.

    taxonomy_labels.json and regime_counts.json are always generated.
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    regime_counts = build_regime_counts(token_records)
    taxonomy_labels = build_taxonomy_labels(token_records)

    rejections_by_gate = Counter(rec.get("primary_rejection_gate", "other") for rec in token_records)
    sources = Counter(rec.get("data_source", "unknown") for rec in token_records)
    survivors = sum(1 for rec in token_records if rec.get("engine_verdict") == "survivor")

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_commit": _git_commit(),
        "methodology": {
            "regime_gate": config.get("regime_gate", {}),
            "profitable_definition": "oos_score > 0 per regime",
            "history_days": config.get("data", {}).get("history_days", 270),
            "coarse_sample_size": config.get("grid_search", {}).get("coarse_sample_size"),
        },
        "token_count": len(token_records),
        "sources": dict(sorted(sources.items())),
        "survivors": survivors,
        "rejections_by_gate": dict(sorted(rejections_by_gate.items())),
        "seed_agreement_rate": (
            taxonomy_labels["seed_agreement_counts"].get("agrees", 0) / max(len(token_records), 1)
        ),
        "backfill_manifest_updated_at": (backfill_manifest or {}).get("updated_at"),
        "batch_manifest": batch_manifest,
    }

    write_jsonl(export_path / "dataset.jsonl", token_records)
    write_jsonl(export_path / "rejections.jsonl", rejection_rows)
    write_json(export_path / "manifest.json", manifest)
    write_json(export_path / "regime_counts.json", regime_counts)
    write_json(export_path / "taxonomy_labels.json", taxonomy_labels)

    return export_path