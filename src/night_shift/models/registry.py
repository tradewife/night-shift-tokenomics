"""Design model registry."""

from night_shift.models.base import DesignModel
from night_shift.models.tokenomics_design import TokenomicsDesign

def get_model(name: str = "tokenomics_mvp", use_stub: bool = False) -> DesignModel:
    if name == "tokenomics_mvp":
        return TokenomicsDesign(use_stub=use_stub)
    raise KeyError(f"Unknown design model: {name}. Available: ['tokenomics_mvp']")