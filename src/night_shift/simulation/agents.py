"""Agent types for Stage 5 economic simulation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class AgentRole(Enum):
    RATIONAL_HOLDER = "rational_holder"
    SPECULATOR = "speculator"
    WHALE = "whale"
    TREASURY = "treasury"
    ATTACKER = "attacker"


@dataclass
class MarketState:
    """Token economy state at one simulation timestep."""

    day: int
    price: float
    treasury: float
    circulating_supply: float
    holder_count: int
    top10_holder_pct: float
    holder_value_index: float
    daily_volume: float
    death_spiral: bool = False


@dataclass
class ScenarioConfig:
    """Per-scenario stress modifiers."""

    name: str
    whale_dump_prob: float = 0.02
    whale_dump_size: float = 0.05
    attack_prob: float = 0.01
    treasury_drain_pct: float = 0.0
    sell_pressure_mult: float = 1.0
    liquidity_crunch: float = 0.0
    narrative_shock: float = 0.0


SCENARIOS: Dict[str, ScenarioConfig] = {
    "baseline": ScenarioConfig(name="baseline"),
    "sell_pressure": ScenarioConfig(
        name="sell_pressure",
        whale_dump_prob=0.30,
        whale_dump_size=0.15,
        sell_pressure_mult=2.0,
    ),
    "governance_attack": ScenarioConfig(
        name="governance_attack",
        attack_prob=0.40,
        treasury_drain_pct=0.25,
    ),
    "liquidity_crunch": ScenarioConfig(
        name="liquidity_crunch",
        liquidity_crunch=0.35,
        sell_pressure_mult=1.5,
    ),
    "narrative_shock": ScenarioConfig(
        name="narrative_shock",
        narrative_shock=0.30,
        sell_pressure_mult=1.8,
    ),
}


def treasury_control_resistance(params: Dict[str, Any]) -> float:
    """How resistant treasury is to drain attempts (0–1)."""
    control = params.get("treasury_control", "governance")
    delay = min(params.get("execution_delay_days", 0) / 7, 1.0)
    threshold = min(params.get("proposal_threshold_pct", 10) / 20, 1.0)
    base = {"timelock": 0.85, "autonomous": 0.70, "multisig": 0.55, "governance": 0.35}.get(
        control, 0.3
    )
    return min(1.0, base + delay * 0.1 + threshold * 0.05)