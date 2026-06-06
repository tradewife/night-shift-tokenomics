"""Five-stage tokenomics research pipeline orchestrator."""

import time
from typing import Dict, List, Optional

from night_shift.bridge.security import load_security_risk_feed
from night_shift.config.loader import load_config
from night_shift.core.logging import log
from night_shift.core.types import CandidateResult
from night_shift.data.loader import load_token_data
from night_shift.models.registry import get_model
from night_shift.reporting.report import generate_report
from night_shift.scoring.resilience import score_candidates
from night_shift.search.darwinian import darwinian_evolution
from night_shift.search.grid import coarse_grid_search, fine_refinement, run_experiments
from night_shift.reporting.artifacts import get_run_dir
from night_shift.validation.evaluate import evaluate_candidate
from night_shift.validation.folds import create_folds
from night_shift.validation.agent_stress import run_agent_stress_phase
from night_shift.validation.robustness import run_robustness_phase


def run_night_shift(
    config_path: Optional[str] = None,
    tokens: Optional[List[str]] = None,
    dry_run: Optional[bool] = None,
    use_seed_dataset: bool = False,
    seed_limit: Optional[int] = None,
    fetch_fresh: bool = False,
) -> Dict:
    """
    Main entry point for the Night Shift Tokenomics research pipeline.

    Stages:
      1. Data load (synthetic / cache / Helius / bootstrap)
      2. Walk-forward fold creation
      3. Coarse grid + fine refinement
      4. Darwinian evolution
      5. Morning report + Resilience Scores
    """
    start_time = time.time()
    config = load_config(config_path)

    token_list = tokens or config.get("tokens")
    is_dry_run = dry_run if dry_run is not None else config.get("dry_run", True)
    data_config = config.get("data", {})
    wfa_config = config.get("wfa", {})
    of_config = {
        **config.get("overfitting", {}),
        "regime_gate": config.get("regime_gate", {}),
    }
    grid_config = config.get("grid_search", {})
    model = get_model(config.get("model", "tokenomics_mvp"), use_stub=is_dry_run)

    log("=" * 70)
    log("NIGHT SHIFT TOKENOMICS — Research Pipeline")
    if use_seed_dataset or data_config.get("use_seed_dataset"):
        log(f"Dataset: seed manifest (limit={seed_limit or data_config.get('seed_limit', 'all')})")
    elif token_list:
        log(f"Tokens: {', '.join(token_list)}")
    log(f"Model: {model.name}")
    log(f"Mode: {'dry run (stub evaluator)' if is_dry_run else 'live (real simulator)'}")
    log("=" * 70)

    # Stage 1: Data
    log("\n── Stage 1: Data ──")
    token_data = load_token_data(
        tokens=token_list,
        dry_run=is_dry_run,
        fetch_fresh=fetch_fresh or data_config.get("fetch_fresh", False),
        use_seed_dataset=use_seed_dataset or data_config.get("use_seed_dataset", False),
        seed_limit=seed_limit or data_config.get("seed_limit"),
        history_days=data_config.get("history_days", 180),
        cache_dir=data_config.get("cache_dir"),
        manifest_path=data_config.get("seed_manifest"),
        data_config=data_config,
    )
    if not token_data:
        raise RuntimeError("No token data loaded")

    sources = {k: v.attrs.get("source", "?") for k, v in token_data.items()}
    labels = {k: v.attrs.get("label", "?") for k, v in token_data.items()}
    log(f"  Loaded {len(token_data)} tokens — sources: {sources}")

    # Stage 2: Walk-forward folds
    log("\n── Stage 2: Walk-Forward Folds ──")
    min_bars = min(len(df) for df in token_data.values())
    folds = create_folds(
        total_bars=min_bars,
        num_folds=wfa_config.get("num_folds", 6),
        test_fold_days=wfa_config.get("test_fold_days", 30),
        warmup_bars=wfa_config.get("warmup_bars", 30),
        bars_per_day=wfa_config.get("bars_per_day", 1),
    )
    log(f"Created {len(folds)} folds from {min_bars} daily bars")
    for fold in folds:
        log(
            f"  Fold {fold.fold_num}: train=[{fold.train_start_idx}:{fold.train_end_idx}] "
            f"test=[{fold.test_start_idx}:{fold.test_end_idx}]"
        )

    all_results: Dict[str, List[CandidateResult]] = {}

    # Stage 2b: Baseline
    log("\n── Stage 2b: Default Design Baseline ──")
    for token, events in token_data.items():
        baseline = evaluate_candidate(
            events,
            folds,
            model.default_params(),
            token,
            model,
            of_config,
            compute_fragility=True,
        )
        all_results[token] = [baseline]
        log(
            f"  {token} ({labels.get(token)}): OOS score={baseline.oos_score:+.3f} "
            f"consistency={baseline.oos_consistency:.0%} "
            f"survivor={baseline.survivor_score:.3f}"
        )

    # Stage 3: Coarse grid
    if grid_config.get("coarse", True):
        log("\n── Stage 3: Coarse Grid Search ──")
        for token, events in token_data.items():
            coarse_results = coarse_grid_search(
                events,
                folds,
                token,
                model,
                of_config,
                coarse_window_days=grid_config.get("coarse_window_days", 30),
                bars_per_day=wfa_config.get("bars_per_day", 1),
            )
            all_results[token].extend(coarse_results)

    # Stage 3b: Fine refinement
    log("\n── Stage 3b: Fine Refinement ──")
    top_n = grid_config.get("fine_refinement_top_n", 20)
    for token, events in token_data.items():
        top_candidates = sorted(
            all_results[token],
            key=lambda r: r.survivor_score,
            reverse=True,
        )[:top_n]
        fine_results = fine_refinement(
            events, folds, token, model, top_candidates, of_config
        )
        all_results[token].extend(fine_results)

    # Stage 4: Darwinian
    log("\n── Stage 4: Darwinian Evolution ──")
    darwinian_config = {
        "darwinian_generations": grid_config.get("darwinian_generations", 3),
        "darwinian_population": grid_config.get("darwinian_population", 20),
        "perturbation_range": (0.05, 0.15),
        "darwinian_offspring": 3,
    }
    for token, events in token_data.items():
        evolved = darwinian_evolution(
            events,
            folds,
            token,
            model,
            all_results[token],
            of_config,
            darwinian_config,
        )
        all_results[token].extend(evolved)

    # Stage 4c: Experiments
    experiments = config.get("experiments", [])
    if experiments:
        log(f"\n── Stage 4c: Custom Experiments ({len(experiments)}) ──")
        for token, events in token_data.items():
            exp_results = run_experiments(
                events, folds, token, model, experiments, of_config
            )
            all_results[token].extend(exp_results)

    # Stage 4d: Robustness gates (Monte Carlo + CPCV/PBO + sensitivity)
    robustness_results: Dict = {}
    if config.get("robustness", {}).get("enabled", False):
        log("\n── Stage 4d: Robustness Gates ──")
        run_dir = get_run_dir(config.get("output_dir"))
        robustness_results = run_robustness_phase(
            token_data,
            all_results,
            model,
            folds,
            config,
            output_dir=run_dir / "robustness",
        )
        passed = sum(
            1
            for token_results in robustness_results.values()
            for r in token_results
            if r.get("verdict", {}).get("passed")
        )
        total = sum(len(v) for v in robustness_results.values())
        log(f"  Robustness: {passed}/{total} candidates passed all gates")

    # Stage 5: Agent stress simulation (attack surface gate)
    agent_stress_results: Dict = {}
    stress_by_params: Dict = {}
    if config.get("agent_simulation", {}).get("enabled", False):
        log("\n── Stage 5: Agent Stress Simulation ──")
        run_dir = get_run_dir(config.get("output_dir"))
        agent_stress_results = run_agent_stress_phase(
            token_data,
            all_results,
            config,
            output_dir=run_dir / "agent_stress",
        )
        stress_by_params = agent_stress_results.pop("_by_params", {})
        passed = sum(
            1
            for token_results in agent_stress_results.values()
            if isinstance(token_results, list)
            for r in token_results
            if r.get("passed")
        )
        total = sum(
            len(v) for k, v in agent_stress_results.items() if k != "_by_params"
        )
        log(f"  Agent stress: {passed}/{total} candidates passed attack surface gates")

    # Resilience scoring (optionally penalized by Security-track risk feed)
    log("\n── Stage 5b: Resilience Scoring ──")
    security_cfg = config.get("security_bridge", {})
    security_feed = None
    if security_cfg.get("enabled", False):
        security_feed = load_security_risk_feed(security_cfg.get("risk_feed_path"))
        if security_feed:
            log(f"  Security bridge: {security_feed.get('findings_count', 0)} findings loaded")
        else:
            log("  Security bridge: enabled but risk feed not found")

    resilience_rankings = {}
    for token, results in all_results.items():
        survivors = [r for r in results if not r.rejected]
        if survivors:
            top_survivors = sorted(survivors, key=lambda r: r.survivor_score, reverse=True)[:5]
            resilience_rankings[token] = score_candidates(
                top_survivors,
                security_feed=security_feed,
                agent_stress_by_params=stress_by_params or None,
            )
            best_r = resilience_rankings[token][0]
            log(f"  {token}: top resilience={best_r['resilience_score']:.1f}/100")

    # Stage 6: Report
    log("\n── Stage 6: Morning Report ──")
    run_seconds = time.time() - start_time
    run_dir = generate_report(
        all_results,
        config,
        run_seconds,
        output_dir=config.get("output_dir"),
        resilience_rankings=resilience_rankings,
        data_sources=sources,
        robustness_results=robustness_results,
        agent_stress_results=agent_stress_results,
        security_feed=security_feed,
    )
    log(f"Report written to {run_dir}")
    log(f"Total runtime: {run_seconds / 60:.1f} minutes")

    survivors = [
        r
        for results in all_results.values()
        for r in results
        if not r.rejected
    ]
    best = max(survivors, key=lambda r: r.survivor_score) if survivors else None
    if best:
        log(
            f"Best survivor: {best.token} score={best.survivor_score:.3f} "
            f"(OOS={best.oos_score:+.3f}, consistency={best.oos_consistency:.0%})"
        )
    else:
        log("No survivors passed all gates")

    return {
        "run_dir": str(run_dir),
        "runtime_seconds": run_seconds,
        "results": all_results,
        "resilience_rankings": resilience_rankings,
        "robustness_results": robustness_results,
        "agent_stress_results": agent_stress_results,
        "best": best,
    }