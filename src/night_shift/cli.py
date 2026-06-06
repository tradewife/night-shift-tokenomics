"""CLI entry point for Night Shift Tokenomics."""

import argparse
import sys

from night_shift.core.pipeline import run_night_shift
from night_shift.data.loader import prefetch_seed_dataset


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
        help="Token symbols or mints to evaluate (overrides config)",
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
        help="Use real simulator with cached/bootstrapped/Helius data",
    )
    parser.add_argument(
        "--seed-dataset",
        action="store_true",
        help="Evaluate all tokens from data/seed_tokens.json",
    )
    parser.add_argument(
        "--seed-limit",
        type=int,
        default=None,
        help="Limit number of seed tokens to evaluate",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Force fresh fetch from Helius (requires HELIUS_API_KEY)",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Prefetch and cache seed token data, then exit",
    )

    args = parser.parse_args()

    dry_run = None
    if args.dry_run:
        dry_run = True
    elif args.no_dry_run:
        dry_run = False

    try:
        if args.prefetch:
            sources = prefetch_seed_dataset(
                limit=args.seed_limit,
                fetch_fresh=args.fetch or True,
            )
            print(f"Prefetched {len(sources)} tokens:")
            for symbol, source in sources.items():
                print(f"  {symbol}: {source}")
            return

        result = run_night_shift(
            config_path=args.config,
            tokens=args.tokens,
            dry_run=dry_run,
            use_seed_dataset=args.seed_dataset,
            seed_limit=args.seed_limit,
            fetch_fresh=args.fetch,
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