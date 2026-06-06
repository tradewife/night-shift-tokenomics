"""Tests for Security → Tokenomics cross-track bridge."""

import json
from pathlib import Path

from night_shift.bridge.security import (
    compute_security_penalty,
    load_security_risk_feed,
    match_risk_patterns,
)
from night_shift.core.types import CandidateResult
from night_shift.scoring.resilience import compute_resilience_score


def _sample_risk_feed() -> dict:
    return {
        "schema_version": "1.0",
        "source": "night-shift-security",
        "findings_count": 2,
        "risk_patterns": [
            {
                "pattern_id": "governance_capture_risk",
                "template_id": "governance_capture",
                "attack_surface": "governance",
                "penalty": 30,
                "triggers": {
                    "proposal_threshold_pct_max": 40,
                    "execution_delay_days_max": 2,
                    "treasury_control": ["governance", "multisig"],
                },
            },
            {
                "pattern_id": "treasury_drain_risk",
                "template_id": "treasury_drain",
                "attack_surface": "treasury",
                "penalty": 20,
                "triggers": {
                    "treasury_control": ["governance", "multisig", "autonomous"],
                    "treasury_pct_min": 30,
                },
            },
        ],
    }


def test_match_risk_patterns_governance_design():
    params = {
        "proposal_threshold_pct": 10,
        "execution_delay_days": 1,
        "treasury_control": "governance",
        "treasury_pct": 40,
    }
    matched = match_risk_patterns(params, _sample_risk_feed())
    templates = {p["template_id"] for p in matched}
    assert "governance_capture" in templates
    assert "treasury_drain" in templates


def test_compute_security_penalty_caps_at_40():
    params = {
        "proposal_threshold_pct": 5,
        "execution_delay_days": 0,
        "treasury_control": "governance",
        "treasury_pct": 50,
    }
    penalty, explain = compute_security_penalty(params, _sample_risk_feed())
    assert penalty == 40.0
    assert "security_governance" in explain


def test_resilience_score_reduced_with_security_feed():
    candidate = CandidateResult(
        token="TEST",
        params={
            "proposal_threshold_pct": 10,
            "execution_delay_days": 1,
            "treasury_control": "governance",
            "treasury_pct": 40,
            "holder_pct": 25,
            "burn_pct": 10,
            "dev_pct": 10,
        },
        oos_score=0.5,
        oos_pnl=100.0,
        oos_pf=1.2,
        oos_wr=0.55,
        oos_max_dd=20,
        oos_consistency=0.6,
        oos_avg_events_per_fold=10,
        oos_mean_duration=5.0,
        oos_outcomes={},
        is_score=0.6,
        is_pnl=120.0,
        overfitting_score=0.2,
        fragility=0.1,
        survivor_score=0.7,
    )
    without = compute_resilience_score(candidate)
    with_feed = compute_resilience_score(candidate, security_feed=_sample_risk_feed())
    assert with_feed.attack_resistance < without.attack_resistance
    assert with_feed.total <= without.total
    assert any(k.startswith("security_") for k in with_feed.explainability)


def test_load_security_risk_feed(tmp_path: Path):
    feed_path = tmp_path / "tokenomics_risk_feed.json"
    feed_path.write_text(json.dumps(_sample_risk_feed()))
    loaded = load_security_risk_feed(feed_path)
    assert loaded is not None
    assert loaded["findings_count"] == 2
    assert load_security_risk_feed(tmp_path / "missing.json") is None