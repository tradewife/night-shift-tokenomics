"""Resilience Score components (SPEC §4) from simulation outputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from night_shift.bridge.security import compute_security_penalty
from night_shift.core.types import CandidateResult


@dataclass
class ResilienceComponents:
    value_accrual: float = 0.0
    holder_retention: float = 0.0
    attack_resistance_base: float = 0.0
    security_penalty: float = 0.0
    attack_resistance: float = 0.0
    stress_sustainability: float = 0.0
    agent_attack_surface: float = 0.0
    transparency: float = 0.0
    extraction_penalty: float = 0.0
    total: float = 0.0
    explainability: Dict[str, str] = field(default_factory=dict)


def _attack_resistance_base(params: Dict[str, Any]) -> float:
    treasury_control = params.get("treasury_control", "governance")
    return min(
        100,
        max(
            0,
            params.get("proposal_threshold_pct", 0) * 2
            + params.get("execution_delay_days", 0) * 3
            + {"timelock": 15, "autonomous": 10, "multisig": 8, "governance": 3}.get(
                treasury_control, 0
            ),
        ),
    )


def compute_resilience_score(
    candidate: CandidateResult,
    security_feed: dict | None = None,
    agent_stress: Optional[Dict[str, Any]] = None,
) -> ResilienceComponents:
    """
    Compute a 0–100 Resilience Score with explainability from candidate metrics.

    attack_resistance_base: design params only (pre-security-penalty)
    security_penalty: cross-track penalty from Night Shift Security feed
    attack_resistance: base minus security_penalty (used in total)
    agent_attack_surface: aggregate score from Stage 5 agent stress sim
    """
    params = candidate.params
    outcomes = candidate.oos_outcomes

    holder_pct = params.get("holder_pct", 0)
    burn_pct = params.get("burn_pct", 0)
    dev_pct = params.get("dev_pct", 0)
    treasury_control = params.get("treasury_control", "governance")

    value_accrual = min(100, max(0, candidate.oos_score * 10 + holder_pct * 0.5 + burn_pct * 0.3))
    holder_retention = min(
        100,
        max(0, candidate.oos_consistency * 100 + outcomes.get("holder_growth_pct", 0) * 0.5),
    )

    attack_base = _attack_resistance_base(params)
    security_penalty, security_explain = compute_security_penalty(params, security_feed)
    attack_final = max(0, attack_base - security_penalty)

    stress_sustainability = min(
        100,
        max(0, (1 - candidate.oos_max_dd / 100) * 60 + candidate.oos_consistency * 40),
    )

    agent_attack_surface = 0.0
    if agent_stress:
        agent_attack_surface = float(agent_stress.get("aggregate_attack_surface", 0))
        stress_sustainability = round(
            0.5 * stress_sustainability + 0.5 * agent_attack_surface, 1
        )

    transparency = min(
        100,
        max(0, {"autonomous": 90, "timelock": 85, "multisig": 70, "governance": 50}.get(
            treasury_control, 40
        )),
    )
    extraction_penalty = min(50, dev_pct * 2 + outcomes.get("concentration_leak", 0) * 5)

    total = max(
        0,
        min(
            100,
            (
                value_accrual * 0.25
                + holder_retention * 0.20
                + attack_final * 0.20
                + stress_sustainability * 0.20
                + transparency * 0.15
            )
            - extraction_penalty,
        ),
    )

    explainability: Dict[str, str] = {}
    if holder_pct >= 25:
        explainability["holder_pct"] = f"+{holder_pct}% to holders boosts value accrual"
    if dev_pct > 15:
        explainability["dev_pct"] = f"{dev_pct}% dev allocation increases extraction penalty"
    if candidate.oos_consistency < 0.5:
        explainability["consistency"] = "Low cross-fold consistency suggests fragility"
    if candidate.overfitting_score > 0.4:
        explainability["overfitting"] = "High IS/OOS gap indicates narrative overfitting risk"
    if security_penalty > 0:
        explainability["security_penalty"] = (
            f"Security feed penalty: -{security_penalty:.0f} attack resistance"
        )
    if agent_attack_surface > 0:
        explainability["agent_stress"] = (
            f"Agent stress sim attack surface: {agent_attack_surface:.0f}/100"
        )
    explainability.update(security_explain)

    return ResilienceComponents(
        value_accrual=round(value_accrual, 1),
        holder_retention=round(holder_retention, 1),
        attack_resistance_base=round(attack_base, 1),
        security_penalty=round(security_penalty, 1),
        attack_resistance=round(attack_final, 1),
        stress_sustainability=round(stress_sustainability, 1),
        agent_attack_surface=round(agent_attack_surface, 1),
        transparency=round(transparency, 1),
        extraction_penalty=round(extraction_penalty, 1),
        total=round(total, 1),
        explainability=explainability,
    )


def score_candidates(
    candidates: List[CandidateResult],
    security_feed: dict | None = None,
    agent_stress_by_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Attach resilience scores to a list of candidates."""
    scored = []
    for candidate in candidates:
        agent_stress = None
        if agent_stress_by_params:
            key = str(sorted(candidate.params.items()))
            agent_stress = agent_stress_by_params.get(key)
        components = compute_resilience_score(
            candidate,
            security_feed=security_feed,
            agent_stress=agent_stress,
        )
        scored.append(
            {
                "token": candidate.token,
                "survivor_score": candidate.survivor_score,
                "resilience_score": components.total,
                "components": components,
                "params": candidate.params,
            }
        )
    return sorted(scored, key=lambda x: x["resilience_score"], reverse=True)