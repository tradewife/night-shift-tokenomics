"""Base contract for tokenomic design models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class WindowMetrics:
    """Evaluation metrics for one train or test window."""

    score: float
    pnl: float
    profit_factor: float
    win_rate: float
    max_drawdown_pct: float
    event_count: int
    avg_duration: float
    outcomes: Dict[str, int] = field(default_factory=dict)


class DesignModel(ABC):
    """Abstract tokenomic design model evaluated by the research pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this model encodes."""

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Default parameter values."""

    @abstractmethod
    def param_grid(self, phase: str = "coarse") -> Dict[str, List[Any]]:
        """
        Return searchable parameter grid.

        phase: "coarse" for fast screening, "fine" for refinement sweep.
        """

    @abstractmethod
    def simulate_window(
        self,
        events: pd.DataFrame,
        params: Dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> WindowMetrics:
        """Evaluate design params on a slice of token event history."""

    def is_discrete(self, param_name: str) -> bool:
        """Whether a parameter should be skipped during Darwinian perturbation."""
        return param_name in self.discrete_params()

    def discrete_params(self) -> set[str]:
        """Parameter names that are categorical and should not be perturbed."""
        return set()

    def param_floor(self, param_name: str) -> float:
        """Minimum value for a numeric parameter after perturbation."""
        return 0.01

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate params before evaluation."""
        return dict(params)