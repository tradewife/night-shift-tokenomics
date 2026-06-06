"""MVP tokenomics design model combining four primitive groups."""

from typing import Any, Dict, List

import pandas as pd

from night_shift.models.base import DesignModel, WindowMetrics
from night_shift.simulation.fast_evaluator import evaluate_design_stub
from night_shift.simulation.tokenomics_sim import evaluate_design
from night_shift.taxonomy.parameter_space import (
    COARSE_GRID,
    DEFAULT_PARAMS,
    DISCRETE_PARAMS,
    FINE_GRID,
    PARAM_FLOORS,
)


class TokenomicsDesign(DesignModel):
    """Composite design model covering fee routing, supply, vesting, and governance."""

    def __init__(self, use_stub: bool = False):
        self._use_stub = use_stub

    @property
    def name(self) -> str:
        return "tokenomics_mvp"

    @property
    def description(self) -> str:
        return "MVP tokenomic design: fee routing, supply mechanics, vesting, governance"

    def default_params(self) -> Dict[str, Any]:
        return dict(DEFAULT_PARAMS)

    def param_grid(self, phase: str = "coarse") -> Dict[str, List[Any]]:
        if phase == "fine":
            return dict(FINE_GRID)
        return dict(COARSE_GRID)

    def discrete_params(self) -> set[str]:
        return set(DISCRETE_PARAMS)

    def param_floor(self, param_name: str) -> float:
        return PARAM_FLOORS.get(param_name, 0.01)

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = {**self.default_params(), **params}
        fee_total = (
            validated.get("treasury_pct", 0)
            + validated.get("holder_pct", 0)
            + validated.get("burn_pct", 0)
            + validated.get("dev_pct", 0)
        )
        if fee_total > 0 and abs(fee_total - 100) > 1:
            scale = 100 / fee_total
            for key in ("treasury_pct", "holder_pct", "burn_pct", "dev_pct"):
                if key in validated and isinstance(validated[key], (int, float)):
                    validated[key] = round(validated[key] * scale, 2)
        return validated

    def simulate_window(
        self,
        events: pd.DataFrame,
        params: Dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> WindowMetrics:
        validated = self.validate_params(params)
        if self._use_stub:
            token = str(events.attrs.get("token", "unknown"))
            return evaluate_design_stub(events, validated, start_idx, end_idx, token=token)
        return evaluate_design(events, validated, start_idx, end_idx)