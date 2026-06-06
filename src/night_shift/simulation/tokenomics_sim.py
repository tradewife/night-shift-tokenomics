"""Real single-window tokenomics simulator using on-chain daily metrics."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from night_shift.models.base import WindowMetrics


BASE_FEE_BPS = 30  # 0.30% proxy on transfer volume


def _governance_shield(params: Dict[str, Any]) -> float:
    threshold = params.get("proposal_threshold_pct", 10)
    delay = params.get("execution_delay_days", 3)
    control = params.get("treasury_control", "autonomous")
    control_bonus = {
        "autonomous": 0.15,
        "timelock": 0.25,
        "multisig": 0.12,
        "governance": 0.05,
    }
    return (
        control_bonus.get(control, 0)
        + min(threshold / 20, 1.0) * 0.15
        + min(delay / 7, 1.0) * 0.10
    )


def _vesting_sell_pressure(params: Dict[str, Any], volume: float) -> float:
    vesting_type = params.get("vesting_type", "linear")
    cliff = params.get("cliff_days", 90)
    if vesting_type == "none":
        return 0.0
    cliff_factor = max(0, 1 - cliff / 365)
    type_mult = {"cliff": 0.8, "linear": 0.5, "milestone": 0.6}.get(vesting_type, 0.4)
    return volume * 0.0004 * type_mult * cliff_factor


def compute_daily_holder_returns(
    window: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute per-day net holder value change under a tokenomic design.

    Returns daily return series (%) and aggregate flow breakdown.
    """
    fee_rate = BASE_FEE_BPS / 10000
    daily_returns = []
    totals = {
        "holder_accrual": 0.0,
        "treasury_capture": 0.0,
        "burn_value": 0.0,
        "dev_extraction": 0.0,
        "inflation_cost": 0.0,
        "concentration_leak": 0.0,
        "vesting_pressure": 0.0,
        "governance_shield": 0.0,
    }

    gov_shield = _governance_shield(params)

    for _, row in window.iterrows():
        volume = float(row.get("transfer_volume", 0))
        if volume <= 0:
            daily_returns.append(0.0)
            continue

        fees = volume * fee_rate
        holder_share = fees * params.get("holder_pct", 0) / 100
        treasury_share = fees * params.get("treasury_pct", 0) / 100
        burn_share = fees * params.get("burn_pct", 0) / 100
        dev_share = fees * params.get("dev_pct", 0) / 100

        inflation_cost = float(row.get("mint_amount", 0)) * params.get("inflation_rate_pct", 0) / 100
        organic_burn = float(row.get("burn_amount", 0))
        buyback = volume * params.get("buyback_pct", 0) / 10000
        concentration = float(row.get("top10_holder_pct", 50))
        conc_leak = max(0, concentration - 35) / 65 * volume * 0.0008
        vesting_pressure = _vesting_sell_pressure(params, volume)
        shield = gov_shield * fees * 0.15

        net_holder = (
            holder_share
            + burn_share * 0.35
            + organic_burn * 0.008
            + buyback * 0.012
            + shield
            - inflation_cost * 0.01
            - dev_share
            - conc_leak
            - vesting_pressure
        )

        daily_ret = net_holder / volume * 100
        daily_returns.append(daily_ret)

        totals["holder_accrual"] += holder_share
        totals["treasury_capture"] += treasury_share
        totals["burn_value"] += burn_share + organic_burn * 0.008
        totals["dev_extraction"] += dev_share
        totals["inflation_cost"] += inflation_cost
        totals["concentration_leak"] += conc_leak
        totals["vesting_pressure"] += vesting_pressure
        totals["governance_shield"] += shield

    return np.array(daily_returns), totals


def evaluate_design(
    events: pd.DataFrame,
    params: Dict[str, Any],
    start_idx: int,
    end_idx: int,
) -> WindowMetrics:
    """
    Evaluate a tokenomic design on a window of daily on-chain metrics.

    Metrics reflect holder value accrual, extraction, concentration risk,
    and sustainability — not synthetic random scores.
    """
    window = events.iloc[start_idx:end_idx]
    if len(window) < 3:
        return WindowMetrics(
            score=0.0,
            pnl=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            max_drawdown_pct=0.0,
            event_count=0,
            avg_duration=0.0,
            outcomes={"insufficient_data": 1},
        )

    daily_returns, flows = compute_daily_holder_returns(window, params)

    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        score = float(np.mean(daily_returns)) if len(daily_returns) else 0.0
    else:
        score = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(len(daily_returns)))

    pnl = float(np.sum(daily_returns))
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns <= 0]
    win_rate = len(wins) / len(daily_returns) if len(daily_returns) else 0.0
    pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and len(wins) > 0 else 0.0

    equity = np.cumsum(daily_returns)
    peak = np.maximum.accumulate(equity)
    max_dd = float(abs(np.min(equity - peak))) if len(equity) else 0.0

    holder_growth = 0.0
    if "holder_count" in window.columns and len(window) >= 2:
        start_h = window["holder_count"].iloc[0]
        end_h = window["holder_count"].iloc[-1]
        if start_h > 0:
            holder_growth = (end_h - start_h) / start_h * 100

    outcomes = {
        "holder_accrual": int(flows["holder_accrual"] > flows["dev_extraction"]),
        "net_positive_days": int(len(wins)),
        "extraction_events": int(len(losses)),
        "burn": int(flows["burn_value"] > 0),
        "concentration_leak": int(flows["concentration_leak"] > flows["holder_accrual"] * 0.5),
        "holder_growth_pct": round(holder_growth, 2),
    }

    return WindowMetrics(
        score=round(score, 4),
        pnl=round(pnl, 2),
        profit_factor=round(float(pf), 3),
        win_rate=round(win_rate, 3),
        max_drawdown_pct=round(max_dd, 2),
        event_count=len(daily_returns),
        avg_duration=round(len(window) / max(len(daily_returns), 1), 2),
        outcomes=outcomes,
    )