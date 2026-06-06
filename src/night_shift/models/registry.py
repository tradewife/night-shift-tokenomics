"""Design model registry."""

from typing import Dict

from night_shift.models.base import DesignModel
from night_shift.models.tokenomics_design import TokenomicsDesign

MODELS: Dict[str, DesignModel] = {
    "tokenomics_mvp": TokenomicsDesign(),
}


def get_model(name: str = "tokenomics_mvp") -> DesignModel:
    if name not in MODELS:
        raise KeyError(f"Unknown design model: {name}. Available: {list(MODELS)}")
    return MODELS[name]