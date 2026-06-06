"""Morning report generation — markdown + JSON."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from night_shift.core.types import CandidateResult
from night_shift.reporting.artifacts import get_run_dir
from night_shift.scoring.resilience import compute_resilience_score


def _top_candidates(all_results: Dict[str, List[CandidateResult]], n: int = 10) -> List[CandidateResult]:
    flat = [r for results in all_results.values() for r in results if not r.rejected]
    return sorted(flat, key=lambda r: r.survivor_score, reverse=True)[:n]


def generate_report(
    all_results: Dict[str, List[CandidateResult]],
    config: Dict,
    run_seconds: float,
    output_dir: str | Path | None = None,
    resilience_rankings: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    data_sources: Optional[Dict[str, str]] = None,
) -> Path:
    """Write markdown report and JSON summary to the run directory."""
    run_dir = get_run_dir(output_dir)
    top = _top_candidates(all_results)

    lines = [
        "# Night Shift Tokenomics — Morning Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Runtime:** {run_seconds / 60:.1f} minutes",
        f"**Mode:** {'dry run (stub evaluator)' if config.get('dry_run', True) else 'live (real simulator)'}",
        "",
        "## Summary",
        "",
        f"- Tokens evaluated: {len(all_results)}",
        f"- Total candidates: {sum(len(v) for v in all_results.values())}",
        f"- Survivors (passed gates): {sum(1 for v in all_results.values() for r in v if not r.rejected)}",
        f"- Top survivor score: {top[0].survivor_score:.3f}" if top else "- No survivors",
        "",
    ]

    if data_sources:
        lines.extend(["## Data Sources", ""])
        for token, source in data_sources.items():
            lines.append(f"- **{token}**: {source}")
        lines.append("")

    lines.extend(["## Top Candidates", ""])

    for i, candidate in enumerate(top, 1):
        resilience = compute_resilience_score(candidate)
        lines.extend(
            [
                f"### #{i}: {candidate.token} (Survivor: {candidate.survivor_score:.3f})",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Resilience Score | {resilience.total:.1f}/100 |",
                f"| OOS Score | {candidate.oos_score:+.3f} |",
                f"| OOS Consistency | {candidate.oos_consistency:.0%} |",
                f"| OOS PnL | {candidate.oos_pnl:+.2f}% |",
                f"| Max DD | {candidate.oos_max_dd:.1f}% |",
                f"| Overfitting | {candidate.overfitting_score:.2f} |",
                f"| Fragility | {candidate.fragility:.2f} |",
                "",
            ]
        )
        if resilience.explainability:
            lines.append("**Explainability:**")
            for key, note in resilience.explainability.items():
                lines.append(f"- {note}")
            lines.append("")
        lines.extend(
            [
                "<details><summary>Parameters</summary>",
                "",
                "```json",
                json.dumps(candidate.params, indent=2),
                "```",
                "",
                "</details>",
                "",
            ]
        )

    if resilience_rankings:
        lines.extend(["## Resilience Rankings by Token", ""])
        for token, rankings in resilience_rankings.items():
            if rankings:
                best = rankings[0]
                lines.append(
                    f"- **{token}**: {best['resilience_score']:.1f}/100 "
                    f"(survivor={best['survivor_score']:.3f})"
                )
        lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": run_seconds,
        "dry_run": config.get("dry_run", True),
        "tokens": list(all_results.keys()),
        "data_sources": data_sources or {},
        "top_candidates": [
            {
                "rank": i,
                "token": c.token,
                "survivor_score": c.survivor_score,
                "resilience_score": compute_resilience_score(c).total,
                "oos_score": c.oos_score,
                "oos_consistency": c.oos_consistency,
                "params": c.params,
                "rejected": c.rejected,
            }
            for i, c in enumerate(top, 1)
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    full_results = {
        token: [
            {
                "survivor_score": r.survivor_score,
                "resilience_score": compute_resilience_score(r).total,
                "oos_score": r.oos_score,
                "oos_consistency": r.oos_consistency,
                "rejected": r.rejected,
                "rejection_reason": r.rejection_reason,
                "params": r.params,
            }
            for r in sorted(results, key=lambda x: x.survivor_score, reverse=True)[:50]
        ]
        for token, results in all_results.items()
    }
    (run_dir / "full_results.json").write_text(json.dumps(full_results, indent=2))

    if resilience_rankings:
        resilience_out = {
            token: [
                {
                    "resilience_score": r["resilience_score"],
                    "survivor_score": r["survivor_score"],
                    "components": {
                        "value_accrual": r["components"].value_accrual,
                        "holder_retention": r["components"].holder_retention,
                        "attack_resistance": r["components"].attack_resistance,
                        "stress_sustainability": r["components"].stress_sustainability,
                        "transparency": r["components"].transparency,
                        "extraction_penalty": r["components"].extraction_penalty,
                    },
                    "explainability": r["components"].explainability,
                    "params": r["params"],
                }
                for r in rankings
            ]
            for token, rankings in resilience_rankings.items()
        }
        (run_dir / "resilience_scores.json").write_text(json.dumps(resilience_out, indent=2))

    return run_dir