"""Deterministic stub evaluator for pipeline dry runs."""

import hashlib
from typing import Any, Dict

import numpy as np
import pandas as pd

from night_shift.models.base import WindowMetrics


def _param_seed(params: Dict[str, Any], token: str, window_key: str) -> int:
    payload = f"{token}|{window_key}|" + "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest[:8], 16)


def _fee_routing_score(params: Dict[str, Any]) -> float:
    treasury = params.get("treasury_pct", 30)
    holder = params.get("holder_pct", 25)
    burn = params.get("burn_pct", 10)
    dev = params.get("dev_pct", 10)
    total = treasury + holder + burn + dev
    balance_penalty = abs(total - 100) / 100
    holder_share = holder / max(total, 1)
    burn_share = burn / max(total, 1)
    extractive_penalty = dev / max(total, 1) * 0.5
    return holder_share + burn_share * 0.8 - balance_penalty - extractive_penalty


def _supply_score(params: Dict[str, Any]) -> float:
    inflation = params.get("inflation_rate_pct", 1.0)
    burn_rate = params.get("burn_rate_pct", 0.5)
    buyback = params.get("buyback_pct", 15)
    return (burn_rate + buyback * 0.01) - inflation * 0.15


def _vesting_score(params: Dict[str, Any]) -> float:
    vesting_type = params.get("vesting_type", "linear")
    cliff = params.get("cliff_days", 90)
    type_bonus = {"none": -0.3, "cliff": 0.1, "linear": 0.25, "milestone": 0.2}
    return type_bonus.get(vesting_type, 0) + min(cliff / 365, 0.3)


def _governance_score(params: Dict[str, Any]) -> float:
    threshold = params.get("proposal_threshold_pct", 10)
    delay = params.get("execution_delay_days", 3)
    control = params.get("treasury_control", "autonomous")
    control_bonus = {
        "autonomous": 0.2,
        "timelock": 0.25,
        "multisig": 0.15,
        "governance": 0.05,
    }
    capture_resistance = min(threshold / 20, 1.0) * 0.3 + min(delay / 7, 1.0) * 0.2
    return control_bonus.get(control, 0) + capture_resistance


def evaluate_design_stub(
    events: pd.DataFrame,
    params: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    token: str = "unknown",
) -> WindowMetrics:
    """
    Deterministic pseudo-simulator for wiring the pipeline before real data exists.

    Produces stable, param-sensitive metrics so grid search and Darwinian selection
    produce meaningful rankings during dry runs.
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

    window_key = f"{start_idx}:{end_idx}"
    rng = np.random.default_rng(_param_seed(params, token, window_key))

    base = (
        _fee_routing_score(params)
        + _supply_score(params)
        + _vesting_score(params)
        + _governance_score(params)
    )

    activity = float(window["transfer_volume"].mean()) if "transfer_volume" in window else 1.0
    concentration = float(window["top10_holder_pct"].mean()) if "top10_holder_pct" in window else 50.0
    concentration_penalty = max(0, (concentration - 40) / 60)

    noise = rng.normal(0, 0.05)
    score = max(-2.0, min(5.0, base + activity * 0.02 - concentration_penalty * 0.5 + noise))

    n_events = max(3, int(len(window) * (0.3 + rng.uniform(0, 0.4))))
    returns = rng.normal(score * 0.02, 0.03, n_events)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    pnl = float(returns.sum() * 100)
    pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and len(wins) > 0 else 0.0
    wr = len(wins) / len(returns) if len(returns) else 0.0

    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    max_dd = float(abs(min(equity - peak)) * 100) if len(equity) else 0.0

    outcomes = {
        "holder_accrual": int(n_events * wr),
        "extraction": int(n_events * (1 - wr) * 0.5),
        "burn": int(params.get("burn_pct", 0) / 10),
    }

    return WindowMetrics(
        score=round(score, 4),
        pnl=round(pnl, 2),
        profit_factor=round(pf, 3),
        win_rate=round(wr, 3),
        max_drawdown_pct=round(max_dd, 2),
        event_count=n_events,
        avg_duration=round(len(window) / max(n_events, 1), 2),
        outcomes=outcomes,
    )