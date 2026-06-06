#!/usr/bin/env python3
"""Prefetch Solana token histories into local parquet cache."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from night_shift.data.loader import prefetch_seed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch seed token data into cache")
    parser.add_argument("--limit", type=int, default=None, help="Max tokens to fetch")
    parser.add_argument("--no-fetch", action="store_true", help="Use cache/bootstrap only")
    parser.add_argument("--days", type=int, default=180, help="History days")
    args = parser.parse_args()

    sources = prefetch_seed_dataset(
        limit=args.limit,
        fetch_fresh=not args.no_fetch,
        history_days=args.days,
    )
    print(f"Processed {len(sources)} tokens:")
    for symbol, source in sorted(sources.items()):
        print(f"  {symbol:12s} {source}")


if __name__ == "__main__":
    main()