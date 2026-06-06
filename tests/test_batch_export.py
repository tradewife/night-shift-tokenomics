"""Tests for batch backfill manifest, grid sampling, and Observatory export."""

import json
from pathlib import Path

from night_shift.data.backfill_manifest import (
    load_manifest,
    save_manifest,
    should_skip_token,
    update_token_record,
)
from night_shift.export.observatory import (
    build_regime_counts,
    build_taxonomy_labels,
    export_observatory_artifacts,
)
from night_shift.search.grid import sample_grid_combos
from night_shift.taxonomy.parameter_space import COARSE_GRID


def test_sample_grid_combos_deterministic():
    a = sample_grid_combos(COARSE_GRID, 64, seed=42, token="JUP")
    b = sample_grid_combos(COARSE_GRID, 64, seed=42, token="JUP")
    c = sample_grid_combos(COARSE_GRID, 64, seed=42, token="BONK")
    assert a == b
    assert len(a) == 64
    assert a != c


def test_backfill_manifest_resume_skip(tmp_path: Path):
    manifest = {
        "schema_version": "1.0",
        "history_days": 270,
        "tokens": {},
    }
    update_token_record(
        manifest,
        "JUP",
        {"status": "helius", "source": "helius", "bars": 270},
    )
    path = tmp_path / "backfill_manifest.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)
    assert should_skip_token(loaded, "JUP", min_bars=150)
    assert not should_skip_token(loaded, "JUP", force=True)
    assert not should_skip_token(loaded, "BONK", min_bars=150)


def test_export_always_writes_taxonomy_and_regime_counts(tmp_path: Path):
    token_records = [
        {
            "symbol": "JUP",
            "seed_label": "resilient",
            "seed_category": "defi",
            "data_source": "helius",
            "synthetic_fallback": False,
            "history_days": 270,
            "historical_regime_counts": {"high_vol": 2, "sideways": 2},
            "distinct_regimes": 2,
            "engine_verdict": "rejected",
            "primary_rejection_gate": "regime_gate",
            "rejection_reason": "regime_gate: profitable_regimes=2/3",
            "regime_summary": {
                "gate_passed": False,
                "profitable_regimes": ["high_vol", "sideways"],
                "profitable_count": 2,
            },
            "best_resilience_score": 52.0,
            "taxonomy_signal": "regime_fragile",
            "seed_agreement": "disagrees",
            "confidence": "medium",
        }
    ]
    export_dir = export_observatory_artifacts(
        tmp_path / "v1_test",
        token_records,
        config={"regime_gate": {"enabled": True, "min_profitable_regimes": 3}, "data": {"history_days": 270}},
        rejection_rows=[],
    )
    assert (export_dir / "taxonomy_labels.json").exists()
    assert (export_dir / "regime_counts.json").exists()
    assert (export_dir / "dataset.jsonl").exists()
    assert (export_dir / "manifest.json").exists()

    taxonomy = json.loads((export_dir / "taxonomy_labels.json").read_text())
    regime_counts = json.loads((export_dir / "regime_counts.json").read_text())

    assert "JUP" in taxonomy["tokens"]
    assert taxonomy["tokens"]["JUP"]["taxonomy_signal"] == "regime_fragile"
    assert regime_counts["aggregate"]["tokens_failing_regime_gate"] == 1
    assert regime_counts["per_token"]["JUP"]["engine_profitable"] == ["high_vol", "sideways"]


def test_taxonomy_labels_frozen_rules():
    records = [
        {
            "symbol": "A",
            "seed_label": "resilient",
            "seed_category": "defi",
            "data_source": "helius",
            "synthetic_fallback": False,
            "history_days": 270,
            "historical_regime_counts": {},
            "distinct_regimes": 3,
            "engine_verdict": "survivor",
            "primary_rejection_gate": "none",
            "rejection_reason": "",
            "regime_summary": {"gate_passed": True},
            "best_resilience_score": 65.0,
            "taxonomy_signal": "resilient_pattern",
            "seed_agreement": "agrees",
            "confidence": "high",
        },
        {
            "symbol": "B",
            "seed_label": "fragile",
            "seed_category": "meme",
            "data_source": "bootstrap",
            "synthetic_fallback": True,
            "history_days": 270,
            "historical_regime_counts": {},
            "distinct_regimes": 1,
            "engine_verdict": "rejected",
            "primary_rejection_gate": "regime_gate",
            "rejection_reason": "regime_gate: profitable_regimes=1/3",
            "regime_summary": {"gate_passed": False},
            "best_resilience_score": 30.0,
            "taxonomy_signal": "insufficient_data",
            "seed_agreement": "agrees",
            "confidence": "low",
        },
    ]
    taxonomy = build_taxonomy_labels(records)
    counts = build_regime_counts(records)
    assert taxonomy["signal_counts"]["resilient_pattern"] == 1
    assert taxonomy["signal_counts"]["insufficient_data"] == 1
    assert counts["aggregate"]["token_count"] == 2