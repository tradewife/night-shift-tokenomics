"""Stage 5 agent stress simulation and attack surface gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from night_shift.core.logging import log
from night_shift.core.types import CandidateResult
from night_shift.simulation.agent_engine import run_all_scenarios


@dataclass(frozen=True)
class AgentStressGate:
    """Attack surface thresholds (SPEC §4 gate 4 + Stage 5)."""

    MIN_AGGREGATE_ATTACK_SURFACE: float = 40.0
    MIN_SCENARIO_ATTACK_SURFACE: float = 25.0
    MAX_DEATH_SPIRAL_PROB: float = 0.34
    REJECT_ON_FAIL: bool = True


def _params_key(params: Dict[str, Any]) -> str:
    return str(sorted(params.items()))


def run_agent_stress_phase(
    token_data: Dict[str, pd.DataFrame],
    all_results: Dict[str, List[CandidateResult]],
    config: Dict[str, Any],
    output_dir: Optional[str | Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run agent-based stress scenarios on top survivors that passed robustness.

    Returns per-token stress results keyed for resilience scoring lookup.
    """
    sim_config = config.get("agent_simulation", {})
    if not sim_config.get("enabled", False):
        log("  Agent stress simulation disabled")
        return {}

    gate = AgentStressGate(
        MIN_AGGREGATE_ATTACK_SURFACE=sim_config.get("min_attack_surface", 40.0),
        MIN_SCENARIO_ATTACK_SURFACE=sim_config.get("min_scenario_attack_surface", 25.0),
        MAX_DEATH_SPIRAL_PROB=sim_config.get("max_death_spiral_prob", 0.34),
        REJECT_ON_FAIL=sim_config.get("reject_on_fail", True),
    )

    top_n = sim_config.get("top_n", 3)
    simulation_days = sim_config.get("simulation_days", 90)
    scenarios = sim_config.get(
        "scenarios",
        ["baseline", "sell_pressure", "governance_attack"],
    )
    seed = sim_config.get("seed", 42)

    log(
        f"  Running {len(scenarios)} scenarios × {simulation_days} days "
        f"on top {top_n} survivors per token"
    )

    all_stress: Dict[str, List[Dict[str, Any]]] = {}
    stress_by_key: Dict[str, Dict[str, Any]] = {}
    save_dir = Path(output_dir) if output_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for token, results in all_results.items():
        events = token_data.get(token)
        if events is None or len(events) == 0:
            continue

        survivors = sorted(
            [r for r in results if not r.rejected],
            key=lambda r: r.survivor_score,
            reverse=True,
        )[:top_n]

        if not survivors:
            log(f"    [{token}] No survivors for agent stress")
            continue

        initial_row = events.iloc[-30].to_dict() if len(events) >= 30 else events.iloc[-1].to_dict()
        token_stress: List[Dict[str, Any]] = []

        for i, candidate in enumerate(survivors, 1):
            stress = run_all_scenarios(
                candidate.params,
                scenarios,
                simulation_days=simulation_days,
                seed=seed + hash(token) % 1000 + i,
                initial_row=initial_row,
            )
            passed, failures = _check_gate(stress, gate)
            stress["token"] = token
            stress["survivor_score"] = candidate.survivor_score
            stress["params"] = candidate.params
            stress["passed"] = passed
            stress["gate_failures"] = failures
            stress["verdict"] = _verdict(passed, failures)

            log(
                f"    [{token}] #{i} attack_surface={stress['aggregate_attack_surface']:.0f} "
                f"death_spiral_p={stress['death_spiral_prob']:.0%} — {stress['verdict']}"
            )

            if gate.REJECT_ON_FAIL and not passed:
                candidate.rejected = True
                candidate.rejection_reason = "; ".join(failures) or stress["verdict"]

            token_stress.append(stress)
            stress_by_key[_params_key(candidate.params)] = stress

        all_stress[token] = token_stress
        if save_dir:
            (save_dir / f"{token}_agent_stress.json").write_text(
                json.dumps(token_stress, indent=2, default=str)
            )

    all_stress["_by_params"] = stress_by_key  # type: ignore[assignment]
    return all_stress


def _check_gate(stress: Dict[str, Any], gate: AgentStressGate) -> tuple[bool, List[str]]:
    failures: List[str] = []
    agg = stress.get("aggregate_attack_surface", 0)
    min_scenario = stress.get("min_attack_surface", 0)
    death_p = stress.get("death_spiral_prob", 1)

    if agg < gate.MIN_AGGREGATE_ATTACK_SURFACE:
        failures.append(f"aggregate_attack_surface={agg:.0f}<{gate.MIN_AGGREGATE_ATTACK_SURFACE}")
    if min_scenario < gate.MIN_SCENARIO_ATTACK_SURFACE:
        failures.append(f"min_scenario_attack_surface={min_scenario:.0f}<{gate.MIN_SCENARIO_ATTACK_SURFACE}")
    if death_p > gate.MAX_DEATH_SPIRAL_PROB:
        failures.append(f"death_spiral_prob={death_p:.2f}>{gate.MAX_DEATH_SPIRAL_PROB}")

    return len(failures) == 0, failures


def _verdict(passed: bool, failures: List[str]) -> str:
    if passed:
        return "PASS — Design survives agent stress scenarios"
    if len(failures) == 1:
        return f"FAIL — {failures[0]}"
    return f"FAIL — {len(failures)} attack surface gates failed"