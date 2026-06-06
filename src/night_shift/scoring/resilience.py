"""Resilience Score components (SPEC §4) from simulation outputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from night_shift.core.types import CandidateResult


@dataclass
class ResilienceComponents:
    value_accrual: float = 0.0
    holder_retention: float = 0.0
    attack_resistance: float = 0.0
    stress_sustainability: float = 0.0
    transparency: float = 0.0
    extraction_penalty: float = 0.0
    total: float = 0.0
    explainability: Dict[str, str] = field(default_factory=dict)


def compute_resilience_score(candidate: CandidateResult) -> ResilienceComponents:
    """
    Compute a 0–100 Resilience Score with explainability from candidate metrics.
    """
    params = candidate.params
    outcomes = candidate.oos_outcomes

    holder_pct = params.get("holder_pct", 0)
    burn_pct = params.get("burn_pct", 0)
    dev_pct = params.get("dev_pct", 0)
    treasury_control = params.get("treasury_control", "governance")

    value_accrual = min(100, max(0, candidate.oos_score * 10 + holder_pct * 0.5 + burn_pct * 0.3))
    holder_retention = min(100, max(0, candidate.oos_consistency * 100 + outcomes.get("holder_growth_pct", 0) * 0.5))
    attack_resistance = min(
        100,
        max(
            0,
            params.get("proposal_threshold_pct", 0) * 2
            + params.get("execution_delay_days", 0) * 3
            + {"timelock": 15, "autonomous": 10, "multisig": 8, "governance": 3}.get(treasury_control, 0),
        ),
    )
    stress_sustainability = min(100, max(0, (1 - candidate.oos_max_dd / 100) * 60 + candidate.oos_consistency * 40))
    transparency = min(
        100,
        max(0, {"autonomous": 90, "timelock": 85, "multisig": 70, "governance": 50}.get(treasury_control, 40)),
    )
    extraction_penalty = min(50, dev_pct * 2 + outcomes.get("concentration_leak", 0) * 5)

    total = max(
        0,
        min(
            100,
            (value_accrual * 0.25 + holder_retention * 0.20 + attack_resistance * 0.20
             + stress_sustainability * 0.20 + transparency * 0.15) - extraction_penalty,
        ),
    )

    explainability = {}
    if holder_pct >= 25:
        explainability["holder_pct"] = f"+{holder_pct}% to holders boosts value accrual"
    if dev_pct > 15:
        explainability["dev_pct"] = f"{dev_pct}% dev allocation increases extraction penalty"
    if candidate.oos_consistency < 0.5:
        explainability["consistency"] = "Low cross-fold consistency suggests fragility"
    if candidate.overfitting_score > 0.4:
        explainability["overfitting"] = "High IS/OOS gap indicates narrative overfitting risk"

    return ResilienceComponents(
        value_accrual=round(value_accrual, 1),
        holder_retention=round(holder_retention, 1),
        attack_resistance=round(attack_resistance, 1),
        stress_sustainability=round(stress_sustainability, 1),
        transparency=round(transparency, 1),
        extraction_penalty=round(extraction_penalty, 1),
        total=round(total, 1),
        explainability=explainability,
    )


def score_candidates(candidates: List[CandidateResult]) -> List[Dict[str, Any]]:
    """Attach resilience scores to a list of candidates."""
    scored = []
    for candidate in candidates:
        components = compute_resilience_score(candidate)
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