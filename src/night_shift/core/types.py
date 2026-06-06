"""Domain-agnostic result types for the research pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FoldMetrics:
    """Metrics for one walk-forward fold."""

    fold_num: int
    is_score: float
    oos_score: float
    oos_pnl: float
    oos_pf: float
    oos_wr: float
    oos_max_dd: float
    oos_events: int
    oos_avg_duration: float
    oos_outcomes: Dict[str, int] = field(default_factory=dict)


@dataclass
class CandidateResult:
    """Full walk-forward result for one design candidate on one token."""

    token: str
    params: Dict[str, Any]
    oos_score: float
    oos_pnl: float
    oos_pf: float
    oos_wr: float
    oos_max_dd: float
    oos_consistency: float
    oos_avg_events_per_fold: float
    oos_mean_duration: float
    oos_outcomes: Dict[str, int]
    is_score: float
    is_pnl: float
    overfitting_score: float
    fragility: float
    survivor_score: float
    folds: List[Dict[str, Any]] = field(default_factory=list)
    regime_summary: Dict[str, Any] = field(default_factory=dict)
    rejected: bool = False
    rejection_reason: str = ""
    is_coarse_only: bool = False