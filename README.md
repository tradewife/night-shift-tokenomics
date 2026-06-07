# Night Shift Tokenomics

**The research engine for systematic tokenomics analysis and resilience scoring.**

Night Shift Tokenomics powers the Agentic Tokenomics Observatory by running large-scale, multi-gate simulations of Solana token economic designs.

## What it does

- Explores tokenomic parameter spaces (fee routing, supply mechanics, treasury control, vesting, governance, etc.)
- Applies multi-stage validation: walk-forward analysis, overfitting gates, cross-regime consistency, Monte Carlo stress, and agent simulation
- Produces a **Resilience Score** and taxonomy of resilient vs fragile designs
- Exports Observatory datasets (manifest, dataset JSONL, rejections, regime counts, per-token detail)

## Repository layout

| Path | Purpose |
|------|---------|
| `SPEC.md` | Full technical specification |
| `src/night_shift/` | Pipeline, validation gates, ingestion, batch runner, export |
| `configs/` | `night_config.json` (single runs), `classification_batch.json` (51-token batch) |
| `scripts/` | `run_classification_batch.py`, `fetch_token_data.py` |
| `data/seed_tokens.json` | 51-token seed manifest (tracked) |
| `tests/` | Unit and integration tests |

Runtime artifacts (`data/exports/`, `data/tokens/cache/`, `data/tokens/backfill_manifest.json`, `data/batch_logs/`) are gitignored and stay local.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### Live backfill (Helius)

Set `HELIUS_API_KEY` in your environment (never commit it). Without a key, the pipeline falls back to label-conditioned bootstrap histories.

```bash
export HELIUS_API_KEY=your_key_here

# Phase A: fetch 270-day on-chain histories for all seed tokens
python scripts/run_classification_batch.py --phase backfill --config configs/classification_batch.json

# Phase B: classify and write Observatory export (v2_<date>/)
python scripts/run_classification_batch.py --phase classify --config configs/classification_batch.json
```

Use `--force` to refetch cached tokens. Use `--run-id 2026-06-07` to pin the export directory name.

### Single-token / dev runs

```bash
night-shift --config configs/night_config.json --tokens JUP
```

## Status

**Phase 2 in progress.** Implemented and tested:

- Solana data ingestion (Helius REST, parquet cache, bootstrap fallback)
- Walk-forward grid search with coarse sampling
- Overfitting, cross-regime, robustness, and agent-stress gates
- 51-token batch backfill + classification + Observatory export schema

First public Observatory dataset is pending a full Helius backfill on real on-chain histories. Bootstrap-heavy runs are useful as internal baselines only.

## Related projects

- **Resilient Token Protocol (RTP)** — https://github.com/tradewife/resilient-token-protocol
- **Night Shift Security** (parallel track) — https://github.com/tradewife/night-shift-security
- Website: https://www.resilientprotocol.xyz

## Contact

Kate / tradewife  
X: [@trade_wife](https://x.com/trade_wife)  
GitHub: [tradewife](https://github.com/tradewife)

---

*Built with the "STFU and Build" ethos. Brutal validation over hype.*