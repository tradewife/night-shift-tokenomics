"""CLI entry point for Night Shift Tokenomics."""

import argparse
import sys

from night_shift.core.pipeline import run_night_shift


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Night Shift Tokenomics — systematic tokenomics research engine"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to night_config.json (default: configs/night_config.json)",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=None,
        help="Token identifiers to evaluate (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Use synthetic data and stub evaluator",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Attempt live data ingestion (not yet implemented)",
    )

    args = parser.parse_args()

    dry_run = None
    if args.dry_run:
        dry_run = True
    elif args.no_dry_run:
        dry_run = False

    try:
        result = run_night_shift(
            config_path=args.config,
            tokens=args.tokens,
            dry_run=dry_run,
        )
        print(f"\nDone. Results: {result['run_dir']}", flush=True)
    except NotImplementedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()