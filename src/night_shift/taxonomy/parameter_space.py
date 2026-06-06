"""Formal encoding of tokenomic primitive parameter spaces."""

from typing import Any, Dict, List

# MVP primitive groups per SPEC §3 Stage 1

FEE_ROUTING_PARAMS = {
    "treasury_pct": [20, 30, 40, 50],
    "holder_pct": [10, 20, 30, 40],
    "burn_pct": [0, 5, 10, 20],
    "dev_pct": [5, 10, 15],
}

SUPPLY_PARAMS = {
    "inflation_rate_pct": [0.0, 1.0, 2.0, 5.0],
    "burn_rate_pct": [0.0, 0.5, 1.0, 2.0],
    "buyback_pct": [0, 10, 20, 30],
}

VESTING_PARAMS = {
    "cliff_days": [0, 30, 90, 180],
    "linear_months": [0, 6, 12, 24],
    "vesting_type": ["none", "cliff", "linear", "milestone"],
}

GOVERNANCE_PARAMS = {
    "proposal_threshold_pct": [1, 5, 10, 20],
    "execution_delay_days": [0, 1, 3, 7],
    "treasury_control": ["autonomous", "multisig", "governance", "timelock"],
}

COARSE_GRID: Dict[str, List[Any]] = {
    **{k: v for k, v in FEE_ROUTING_PARAMS.items() if k != "dev_pct"},
    "burn_rate_pct": SUPPLY_PARAMS["burn_rate_pct"],
    "buyback_pct": [0, 20],
    "cliff_days": [0, 90],
    "vesting_type": ["none", "linear"],
    "proposal_threshold_pct": [5, 10],
    "treasury_control": ["autonomous", "governance"],
}

FINE_GRID: Dict[str, List[Any]] = {
    "treasury_pct": [25, 35, 45],
    "holder_pct": [15, 25, 35],
    "burn_pct": [5, 15],
    "inflation_rate_pct": [0.5, 1.5, 3.0],
    "execution_delay_days": [1, 3, 5],
}

DEFAULT_PARAMS: Dict[str, Any] = {
    "treasury_pct": 30,
    "holder_pct": 25,
    "burn_pct": 10,
    "dev_pct": 10,
    "inflation_rate_pct": 1.0,
    "burn_rate_pct": 0.5,
    "buyback_pct": 15,
    "cliff_days": 90,
    "linear_months": 12,
    "vesting_type": "linear",
    "proposal_threshold_pct": 10,
    "execution_delay_days": 3,
    "treasury_control": "autonomous",
}

DISCRETE_PARAMS = {"vesting_type", "treasury_control"}

PARAM_FLOORS = {
    "treasury_pct": 0.0,
    "holder_pct": 0.0,
    "burn_pct": 0.0,
    "dev_pct": 0.0,
    "inflation_rate_pct": 0.0,
    "burn_rate_pct": 0.0,
    "buyback_pct": 0.0,
    "cliff_days": 0,
    "linear_months": 0,
    "proposal_threshold_pct": 0.1,
    "execution_delay_days": 0,
}