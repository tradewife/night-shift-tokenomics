#!/usr/bin/env python3
"""Run Helius backfill + 51-token batch classification for Observatory export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from night_shift.batch.backfill import run_backfill_phase
from night_shift.batch.classify import run_classify_phase
from night_shift.config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Observatory batch backfill + classification")
    parser.add_argument(
        "--config",
        default="configs/classification_batch.json",
        help="Classification config path",
    )
    parser.add_argument(
        "--phase",
        choices=["backfill", "classify", "all"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true", help="Disable resume")
    parser.add_argument("--force", action="store_true", help="Force refetch cached tokens")
    parser.add_argument("--run-id", default=None, help="Export run id (default: UTC date)")
    args = parser.parse_args()

    resume = args.resume and not args.no_resume
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    wfa_cfg = config.get("wfa", {})
    export_base = config.get("export_dir", "data/exports")

    if args.phase in ("backfill", "all"):
        print("=== Phase A: Backfill ===", flush=True)
        summary = run_backfill_phase(
            history_days=data_cfg.get("history_days", 270),
            cache_dir=data_cfg.get("cache_dir"),
            seed_manifest_path=data_cfg.get("seed_manifest"),
            force=args.force,
            resume=resume,
            wfa_config=wfa_cfg,
        )
        print(f"Backfill sources: {summary['sources']}", flush=True)

    if args.phase in ("classify", "all"):
        print("=== Phase B: Classify ===", flush=True)
        result = run_classify_phase(
            config_path=args.config,
            export_base=export_base,
            run_id=args.run_id,
            resume=resume,
            seed_manifest_path=data_cfg.get("seed_manifest"),
        )
        print(
            f"Export: {result['export_dir']} — "
            f"{result['token_count']} tokens, {result['survivors']} survivors",
            flush=True,
        )


if __name__ == "__main__":
    main()