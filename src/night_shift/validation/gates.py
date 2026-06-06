"""Promotion and rejection gates for validated tokenomic designs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResilienceGate:
    """Thresholds a design must clear before proceeding to full simulation."""

    MIN_OOS_SCORE: float = 0.5
    MIN_OOS_FOLDS: int = 3
    MAX_OVERFITTING_RATIO: float = 0.5
    MIN_OOS_CONSISTENCY: float = 0.50
    MAX_DRAWDOWN_PCT: float = 40.0
    MIN_PROFITABLE_REGIMES: int = 2
    MAX_FRAGILITY: float = 0.4
    MIN_RESILIENCE_SCORE: float = 0.3