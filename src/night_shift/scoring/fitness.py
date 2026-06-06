"""Survivor score and overfitting penalty calculations."""

from typing import Dict

from night_shift.config.defaults import SCORE_CAP


def compute_survivor_score(
    oos_score: float,
    oos_consistency: float,
    overfitting_score: float,
    oos_max_dd: float,
    oos_avg_events: float,
    fragility: float,
    of_config: Dict,
) -> float:
    """Primary ranking metric — higher is better."""
    of_penalty = 1.0 - min(overfitting_score, 1.0)
    dd_factor = 1.0 / (1.0 + oos_max_dd / 100)
    min_events = of_config.get("min_events_per_fold", 5)
    event_factor = min(oos_avg_events / max(min_events, 1), 1.0)
    fragility_penalty = 1.0 / (1.0 + fragility)
    return oos_score * oos_consistency * of_penalty * dd_factor * event_factor * fragility_penalty


def winsorize_scores(scores: list[float]) -> list[float]:
    return [max(-SCORE_CAP, min(SCORE_CAP, s)) for s in scores]


def compute_overfitting_score(avg_is_score: float, avg_oos_score: float, skip_is: bool) -> float:
    if skip_is or (avg_is_score == 0 and avg_oos_score == 0):
        return 0.0
    if abs(avg_is_score) > 0.01:
        gap = (avg_is_score - avg_oos_score) / abs(avg_is_score)
        return max(0.0, gap)
    return 0.5 if avg_oos_score < 0 else 0.0