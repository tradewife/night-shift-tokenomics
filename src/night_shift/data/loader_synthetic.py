"""Synthetic token history generator for dry runs."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def generate_synthetic_token_history(
    token: str,
    days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate plausible on-chain token activity for pipeline dry runs."""
    rng = np.random.default_rng(seed + hash(token) % 10000)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    rows = []
    holders = 1000
    top10_pct = 45.0
    for day in range(days):
        ts = start + timedelta(days=day)
        holders += int(rng.integers(-20, 50))
        holders = max(100, holders)
        top10_pct = float(np.clip(top10_pct + rng.normal(0, 1.5), 20, 80))
        volume = float(rng.lognormal(mean=10, sigma=0.5))
        mints = float(rng.uniform(0, volume * 0.01))
        burns = float(rng.uniform(0, volume * 0.005))
        treasury_inflow = float(volume * rng.uniform(0.001, 0.01))

        rows.append(
            {
                "timestamp": ts,
                "day": day,
                "holder_count": holders,
                "top10_holder_pct": top10_pct,
                "transfer_volume": volume,
                "mint_amount": mints,
                "burn_amount": burns,
                "treasury_inflow": treasury_inflow,
                "active_addresses": int(holders * rng.uniform(0.05, 0.2)),
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["token"] = token
    df.attrs["synthetic"] = True
    return df