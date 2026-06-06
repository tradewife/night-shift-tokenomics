"""Promotion, rejection, and robustness gate thresholds."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResilienceGate:
    """Thresholds a design must clear before proceeding to full simulation."""

    MIN_OOS_SCORE: float = 0.5
    MIN_OOS_FOLDS: int = 3
    MAX_OVERFITTING_RATIO: float = 0.5
    MIN_OOS_CONSISTENCY: float = 0.50
    MAX_DRAWDOWN_PCT: float = 40.0
    MIN_PROFITABLE_REGIMES: int = 3
    MAX_FRAGILITY: float = 0.4
    MIN_RESILIENCE_SCORE: float = 0.3


@dataclass(frozen=True)
class RegimeGate:
    """Cross-regime consistency thresholds (SPEC §4 Gate 2)."""

    ENABLED: bool = True
    MIN_PROFITABLE_REGIMES: int = 3
    MIN_REGIME_SCORE: float = 0.0


@dataclass(frozen=True)
class RobustnessGate:
    """Monte Carlo, CPCV/PBO, and sensitivity thresholds (SPEC §4 Stage 4)."""

    MAX_MC_DD_P95: float = 40.0
    MAX_PROB_DD_GT_30: float = 0.25
    MAX_PBO: float = 0.30
    CAUTION_PBO: float = 0.15
    MAX_PARAM_SENSITIVITY: float = 0.4
    MIN_RETURN_P10: float = -50.0