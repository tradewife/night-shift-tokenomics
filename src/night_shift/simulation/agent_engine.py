"""Lightweight agent-based tokenomics simulator (SPEC Stage 5)."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from night_shift.simulation.agents import MarketState, ScenarioConfig, treasury_control_resistance


@dataclass
class AgentSimResult:
    scenario: str
    simulation_days: int
    holder_value_curve: List[float] = field(default_factory=list)
    treasury_curve: List[float] = field(default_factory=list)
    concentration_curve: List[float] = field(default_factory=list)
    death_spiral: bool = False
    sustained_activity: bool = True
    final_holder_value: float = 0.0
    treasury_change_pct: float = 0.0
    attack_surface_score: float = 0.0
    attacks_blocked: int = 0
    attacks_succeeded: int = 0


def _initial_state(events_row: Dict[str, Any], params: Dict[str, Any]) -> MarketState:
    volume = float(events_row.get("transfer_volume", 1e6))
    return MarketState(
        day=0,
        price=1.0,
        treasury=volume * params.get("treasury_pct", 30) / 1000,
        circulating_supply=1_000_000.0,
        holder_count=int(events_row.get("holder_count", 1000)),
        top10_holder_pct=float(events_row.get("top10_holder_pct", 45)),
        holder_value_index=100.0,
        daily_volume=volume,
    )


def _fee_distribution(volume: float, params: Dict[str, Any]) -> Dict[str, float]:
    fees = volume * 0.003
    return {
        "holder": fees * params.get("holder_pct", 0) / 100,
        "treasury": fees * params.get("treasury_pct", 0) / 100,
        "burn": fees * params.get("burn_pct", 0) / 100,
        "dev": fees * params.get("dev_pct", 0) / 100,
    }


def run_agent_simulation(
    params: Dict[str, Any],
    scenario: ScenarioConfig,
    *,
    simulation_days: int = 90,
    seed: int = 42,
    initial_row: Dict[str, Any] | None = None,
) -> AgentSimResult:
    """
    Simulate multi-agent token economy under a stress scenario.

    Agents (implicit): rational holders accrue yield, speculators add volatility,
    whales may dump, attackers attempt treasury/governance exploits, treasury
    agent redistributes per design params.
    """
    rng = np.random.default_rng(seed)
    row = initial_row or {"transfer_volume": 1e6, "holder_count": 1000, "top10_holder_pct": 45}
    state = _initial_state(row, params)
    resistance = treasury_control_resistance(params)

    holder_curve = [state.holder_value_index]
    treasury_curve = [state.treasury]
    concentration_curve = [state.top10_holder_pct]
    attacks_blocked = 0
    attacks_succeeded = 0
    initial_holder_value = state.holder_value_index
    initial_treasury = state.treasury

    inflation_daily = params.get("inflation_rate_pct", 0) / 100 / 365
    burn_daily = params.get("burn_rate_pct", 0) / 100 / 365

    for day in range(1, simulation_days + 1):
        state.day = day
        volume = state.daily_volume * rng.lognormal(0, 0.15) * scenario.sell_pressure_mult
        volume *= max(0.2, 1 - scenario.liquidity_crunch)
        state.daily_volume = volume

        flows = _fee_distribution(volume, params)
        holder_accrual = flows["holder"] + flows["burn"] * 0.3
        state.holder_value_index += holder_accrual / max(volume, 1) * 100
        state.treasury += flows["treasury"]
        state.circulating_supply *= 1 + inflation_daily - burn_daily

        # Rational holders: respond to yield
        if holder_accrual > flows["dev"]:
            state.holder_count += int(rng.integers(1, 8))
            state.top10_holder_pct = max(15, state.top10_holder_pct - 0.1)
        else:
            state.holder_count = max(50, state.holder_count - int(rng.integers(0, 5)))
            state.top10_holder_pct = min(85, state.top10_holder_pct + 0.15)

        # Speculator noise
        state.price *= 1 + rng.normal(0, 0.02 * scenario.sell_pressure_mult)

        # Whale dump
        if rng.random() < scenario.whale_dump_prob:
            dump_impact = scenario.whale_dump_size * scenario.sell_pressure_mult
            state.price *= 1 - dump_impact
            state.holder_value_index -= dump_impact * 50
            state.top10_holder_pct = min(90, state.top10_holder_pct + 3)

        # Governance / treasury attack
        if rng.random() < scenario.attack_prob:
            drain = state.treasury * scenario.treasury_drain_pct
            if rng.random() < resistance:
                attacks_blocked += 1
                drain *= 1 - resistance
            else:
                attacks_succeeded += 1
            state.treasury = max(0, state.treasury - drain)
            state.holder_value_index -= drain / max(state.daily_volume, 1) * 10

        # Narrative shock
        if scenario.narrative_shock > 0:
            state.price *= 1 - scenario.narrative_shock / simulation_days
            state.holder_count = max(30, int(state.holder_count * (1 - scenario.narrative_shock / 200)))

        # Death spiral check
        if state.price < 0.3 or state.holder_value_index < initial_holder_value * 0.4:
            state.death_spiral = True

        holder_curve.append(round(state.holder_value_index, 2))
        treasury_curve.append(round(state.treasury, 2))
        concentration_curve.append(round(state.top10_holder_pct, 2))

    holder_change = (state.holder_value_index - initial_holder_value) / max(initial_holder_value, 1)
    treasury_change = (
        (state.treasury - initial_treasury) / max(initial_treasury, 1) * 100
        if initial_treasury > 0
        else 0
    )
    sustained = state.holder_count > 100 and not state.death_spiral

    # Attack surface score: higher = more resilient under this scenario
    score = 100.0
    if state.death_spiral:
        score -= 40
    if attacks_succeeded > 0:
        score -= min(30, attacks_succeeded * 15)
    if holder_change < 0:
        score -= min(25, abs(holder_change) * 50)
    if state.top10_holder_pct > 60:
        score -= (state.top10_holder_pct - 60) * 0.5
    score = max(0, min(100, score))

    return AgentSimResult(
        scenario=scenario.name,
        simulation_days=simulation_days,
        holder_value_curve=holder_curve,
        treasury_curve=treasury_curve,
        concentration_curve=concentration_curve,
        death_spiral=state.death_spiral,
        sustained_activity=sustained,
        final_holder_value=round(state.holder_value_index, 2),
        treasury_change_pct=round(treasury_change, 2),
        attack_surface_score=round(score, 1),
        attacks_blocked=attacks_blocked,
        attacks_succeeded=attacks_succeeded,
    )


def run_all_scenarios(
    params: Dict[str, Any],
    scenario_names: List[str],
    *,
    simulation_days: int = 90,
    seed: int = 42,
    initial_row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run multiple stress scenarios and aggregate attack surface."""
    from night_shift.simulation.agents import SCENARIOS

    results: List[AgentSimResult] = []
    for i, name in enumerate(scenario_names):
        scenario = SCENARIOS.get(name)
        if not scenario:
            continue
        results.append(
            run_agent_simulation(
                params,
                scenario,
                simulation_days=simulation_days,
                seed=seed + i,
                initial_row=initial_row,
            )
        )

    if not results:
        return {"scenarios": [], "aggregate_attack_surface": 0.0, "death_spiral_prob": 1.0}

    scores = [r.attack_surface_score for r in results]
    death_spirals = sum(1 for r in results if r.death_spiral)
    return {
        "scenarios": [asdict(r) for r in results],
        "aggregate_attack_surface": round(float(np.mean(scores)), 1),
        "min_attack_surface": round(float(np.min(scores)), 1),
        "death_spiral_prob": round(death_spirals / len(results), 3),
        "sustained_all": all(r.sustained_activity for r in results),
    }