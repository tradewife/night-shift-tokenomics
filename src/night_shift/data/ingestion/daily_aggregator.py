"""Aggregate raw transfer events into daily token metrics bars."""

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "holder_count",
    "top10_holder_pct",
    "transfer_volume",
    "mint_amount",
    "burn_amount",
    "treasury_inflow",
    "active_addresses",
]


def aggregate_transfers_to_daily(transfers: List[Dict[str, Any]], days: int = 180) -> pd.DataFrame:
    """
    Aggregate parsed Helius transfer records into daily metric bars.

    Each transfer dict expects: timestamp (unix), amount, type (transfer|mint|burn).
    """
    if not transfers:
        return pd.DataFrame()

    rows = []
    for tx in transfers:
        ts = tx.get("timestamp")
        if ts is None:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(hour=0, minute=0, second=0)
        amount = float(tx.get("amount", 0))
        tx_type = tx.get("type", "transfer")
        rows.append({"date": dt, "amount": amount, "type": tx_type})

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    daily_rows = []
    for date, group in raw.groupby("date"):
        daily_rows.append(
            {
                "timestamp": date,
                "transfer_volume": group.loc[group["type"] == "transfer", "amount"].sum(),
                "mint_amount": group.loc[group["type"] == "mint", "amount"].sum(),
                "burn_amount": group.loc[group["type"] == "burn", "amount"].sum(),
            }
        )
    grouped = pd.DataFrame(daily_rows)
    grouped = _fill_daily_gaps(grouped, days)
    grouped = _estimate_holder_metrics(grouped)
    return grouped


def _fill_daily_gaps(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df["timestamp"].max()
    start = end - pd.Timedelta(days=days - 1)
    full_range = pd.date_range(start=start, end=end, freq="D", tz=timezone.utc)
    filled = (
        df.set_index("timestamp")
        .reindex(full_range, fill_value=0)
        .rename_axis("timestamp")
        .reset_index()
    )
    filled["day"] = np.arange(len(filled))
    return filled


def _estimate_holder_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate holder metrics from volume activity when not directly available."""
    out = df.copy()
    base_holders = 1000
    holders = []
    top10 = []
    active = []
    treasury = []
    h = base_holders
    t10 = 45.0
    for _, row in out.iterrows():
        vol = row.get("transfer_volume", 0)
        h += int(vol / 1e6) - int(row.get("burn_amount", 0) / 2e6)
        h = max(100, h)
        t10 = float(np.clip(t10 + (vol / 1e7) - 0.5, 15, 80))
        holders.append(h)
        top10.append(t10)
        active.append(int(h * min(0.2, vol / 1e7 + 0.03)))
        treasury.append(vol * 0.005)
    out["holder_count"] = holders
    out["top10_holder_pct"] = top10
    out["active_addresses"] = active
    out["treasury_inflow"] = treasury
    return out


def validate_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has required columns and sorted timestamps."""
    if df.empty:
        return df
    out = df.copy()
    if "timestamp" not in out.columns:
        raise ValueError("Daily bars must include timestamp column")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col != "timestamp" else out["timestamp"]
    if "day" not in out.columns:
        out["day"] = np.arange(len(out))
    return out[REQUIRED_COLUMNS + ["day"]]