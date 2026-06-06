"""Tests for Stage 5 agent stress simulation."""

from night_shift.simulation.agent_engine import run_agent_simulation, run_all_scenarios
from night_shift.simulation.agents import SCENARIOS
from night_shift.taxonomy.parameter_space import DEFAULT_PARAMS


def _resilient_params():
    p = dict(DEFAULT_PARAMS)
    p.update(
        {
            "holder_pct": 35,
            "treasury_pct": 35,
            "burn_pct": 15,
            "dev_pct": 5,
            "treasury_control": "timelock",
            "proposal_threshold_pct": 15,
            "execution_delay_days": 5,
        }
    )
    return p


def _fragile_params():
    p = dict(DEFAULT_PARAMS)
    p.update(
        {
            "holder_pct": 5,
            "treasury_pct": 55,
            "burn_pct": 0,
            "dev_pct": 30,
            "treasury_control": "governance",
            "proposal_threshold_pct": 5,
            "execution_delay_days": 0,
        }
    )
    return p


def test_baseline_simulation_runs():
    result = run_agent_simulation(_resilient_params(), SCENARIOS["baseline"], simulation_days=30)
    assert len(result.holder_value_curve) == 31
    assert 0 <= result.attack_surface_score <= 100


def test_resilient_beats_fragile_under_attack():
    resilient = run_all_scenarios(
        _resilient_params(),
        ["sell_pressure", "governance_attack"],
        simulation_days=45,
        seed=1,
    )
    fragile = run_all_scenarios(
        _fragile_params(),
        ["sell_pressure", "governance_attack"],
        simulation_days=45,
        seed=1,
    )
    assert resilient["aggregate_attack_surface"] > fragile["aggregate_attack_surface"]


def test_agent_stress_feeds_resilience_score():
    from night_shift.core.types import CandidateResult
    from night_shift.scoring.resilience import compute_resilience_score

    candidate = CandidateResult(
        token="TEST",
        params=_resilient_params(),
        oos_score=0.8,
        oos_pnl=50,
        oos_pf=1.5,
        oos_wr=0.6,
        oos_max_dd=15,
        oos_consistency=0.7,
        oos_avg_events_per_fold=20,
        oos_mean_duration=3,
        oos_outcomes={},
        is_score=0.7,
        is_pnl=60,
        overfitting_score=0.1,
        fragility=0.1,
        survivor_score=1.0,
    )
    stress = run_all_scenarios(_resilient_params(), ["baseline"], simulation_days=30)
    without = compute_resilience_score(candidate)
    with_stress = compute_resilience_score(candidate, agent_stress=stress)
    assert with_stress.agent_attack_surface > 0
    assert with_stress.stress_sustainability != without.stress_sustainability