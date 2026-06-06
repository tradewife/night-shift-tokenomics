"""Cross-track bridge: consume Night Shift Security risk feed."""

import json
from pathlib import Path
from typing import Any


def load_security_risk_feed(path: str | Path | None) -> dict[str, Any] | None:
    """Load tokenomics_risk_feed.json exported by Night Shift Security."""
    if not path:
        return None
    feed_path = Path(path)
    if not feed_path.exists():
        return None
    with open(feed_path) as f:
        return json.load(f)


def _matches_trigger(params: dict[str, Any], key: str, value: Any) -> bool:
    if key.endswith("_max"):
        base = key[: -len("_max")]
        if base not in params:
            return False
        return float(params[base]) <= float(value)
    if key.endswith("_min"):
        base = key[: -len("_min")]
        if base not in params:
            return False
        return float(params[base]) >= float(value)
    if key not in params:
        return False
    param_val = params[key]
    if isinstance(value, list):
        return param_val in value
    return param_val == value


def match_risk_patterns(params: dict[str, Any], risk_feed: dict[str, Any]) -> list[dict[str, Any]]:
    """Return risk patterns whose triggers match the tokenomics design params."""
    matched: list[dict[str, Any]] = []
    for pattern in risk_feed.get("risk_patterns", []):
        triggers = pattern.get("triggers", {})
        if all(_matches_trigger(params, k, v) for k, v in triggers.items()):
            matched.append(pattern)
    return matched


def compute_security_penalty(
    params: dict[str, Any],
    risk_feed: dict[str, Any] | None,
) -> tuple[float, dict[str, str]]:
    """
    Apply Security-track penalties to attack_resistance.

    Returns (total_penalty, explainability snippets).
    """
    if not risk_feed:
        return 0.0, {}

    matched = match_risk_patterns(params, risk_feed)
    if not matched:
        return 0.0, {}

    total_penalty = min(40.0, sum(p.get("penalty", 0) for p in matched))
    explainability: dict[str, str] = {}
    for pattern in matched:
        surface = pattern.get("attack_surface", pattern.get("template_id", "unknown"))
        explainability[f"security_{surface}"] = (
            f"Security feed: {pattern['template_id']} pattern "
            f"(-{pattern.get('penalty', 0):.0f} attack resistance)"
        )
    return total_penalty, explainability