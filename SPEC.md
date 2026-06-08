# Night Shift Tokenomics — Technical Specification

**Version:** 1.1  
**Date:** 2026-06-08  
**Author:** Grok (for Kate / tradewife)  
**Purpose:** Clone and adapt the existing Night Shift research engine for systematic tokenomics research on Solana. This becomes the core intelligence layer for the Agentic Tokenomics Observatory and Resilience Score. The spec is now also the living source of truth for current implementation status and immediate scoped increments.

---

## 0. Source Code Location & Extraction Instructions (Read This First)

**Critical:** Do not build Night Shift Tokenomics from scratch or from abstract description alone.

The original, working Night Shift implementation (and its supporting architecture) already exists inside the Resilient Token Protocol repository at:

**`/home/kt/projects/rtp/resilient-token-protocol`**

### Instructions for the AI Coding Agent:

1. **Navigate to the directory above** on the local machine.
2. **Explore the project structure** thoroughly, paying special attention to:
   - Any directories or files containing "night_shift", "research", "engine", "simulation", "backtest", "validation", "monte_carlo", "walk_forward", or "darwinian".
   - Python source files that implement parameter search, grid generation, evolutionary algorithms, scoring/fitness functions, or result analysis.
   - Configuration files, YAML/JSON schemas, or dataclasses that define searchable parameter spaces.
   - Data loading, historical windowing, and result persistence logic.
   - Any supporting utilities for parallel execution, logging, or structured output.

3. **Extract and deeply understand** these core reusable components (preserve their structure and philosophy):
   - The overall 5-stage research pipeline architecture
   - Grid / hypothesis space generation logic
   - Walk-forward validation harness (including fold creation and temporal splitting)
   - Darwinian / evolutionary population refinement code
   - Overfitting detection and robustness gates
   - Monte Carlo stress testing framework
   - Scoring / fitness evaluation system
   - Result storage, ranking, and reporting mechanisms

4. **Do NOT copy** trading-specific strategy parameters, perps logic, or Flash Trade integration unless they are genuinely reusable utilities.

5. **Preserve the engineering DNA:**
   - Massive parallel exploration (target 30k+ candidates)
   - Brutal, multi-gate validation before anything is accepted
   - Clear separation between in-sample discovery and out-of-sample validation
   - "Only survivors reach production" mindset

6. After extraction and analysis, adapt the architecture as described in the rest of this specification for the tokenomics domain.

**Note:** There may also be Rust components (on-chain treasury program, Soulguard invariants). These are secondary for the research engine adaptation but may be relevant later for on-chain enforcement of validated designs.

---

## 1. Overview & Goals

Night Shift was originally built as a high-throughput parameter search + validation engine for perps trading strategies. It proved it can:
- Explore 30,000+ configurations per night
- Apply rigorous 9-fold walk-forward validation
- Use Darwinian evolution + Monte Carlo stress testing
- Reject fragile/overfit candidates before they touch capital

**New Mission (Tokenomics edition):**
Turn Night Shift into a **tokenomics research engine** that can:
- Systematically explore and classify token economic designs
- Distinguish resilient, value-accruing designs from extractive or fragile ones
- Generate a reproducible **Resilience Score**
- Power the public Observatory (taxonomy, dataset, explorer)
- Inform the Builder Kit (recommended patterns + anti-patterns)

This directly supports the cyber•Fund / Avalanche / Superteam proposals.

---

## 2. Key Differences from Original Night Shift (Yield Engine)

| Aspect                    | Original (Yield)                  | Tokenomics Edition                     | Notes |
|---------------------------|-----------------------------------|----------------------------------------|-------|
| **Domain**                | Trading strategy parameters       | Tokenomic design parameters            | Core shift |
| **Data**                  | OHLCV + on-chain perps data       | On-chain token events (transfers, supply changes, holder stats, volume) | New data layer needed |
| **Hypothesis Space**      | Indicator params, stops, leverage | Fee rates, burn schedules, treasury allocation, vesting cliffs, redistribution rules, governance thresholds | Much larger combinatorial space |
| **Validation Target**     | PnL, drawdown, Calmar, liquidations | Holder value accrual, retention, attack resistance, sustainability under stress | New fitness function |
| **Simulation**            | Backtester with slippage/fees     | Agent-based economic simulator         | New component |
| **Output**                | Validated trading strategy        | Taxonomy label + Resilience Score + recommended config | Multiple artifacts |
| **Risk of Overfitting**   | Strategy overfitting to history   | Design overfitting to past narratives  | Still critical |

**Philosophy to preserve:**
- Brutal honesty about what works vs what looks good in-sample
- Walk-forward + Monte Carlo as non-negotiable
- "Only survivors reach production" mindset
- Hybrid deterministic + evolutionary search

---

## 3. High-Level Architecture (Keep the 5-Stage Pipeline)

Night Shift Tokenomics should retain the same core loop structure as the original:

```
Stage 1: Grid Search / Hypothesis Generation
    ↓
Stage 2: Walk-Forward Validation (historical on-chain windows)
    ↓
Stage 3: Darwinian Evolution (population refinement)
    ↓
Stage 4: Overfitting / Robustness Gates (3+ gates)
    ↓
Stage 5: Full Agent-Based Simulation (stress scenarios)
    ↓
→ Validated designs + Resilience Scores + Taxonomy labels
```

### Stage 1: Hypothesis Generation (Tokenomic Parameter Space)

Define a structured parameter space covering the key primitives from the Avalanche proposal:

**Core Tokenomic Primitives (to encode as searchable parameters):**
- Fee routing (to treasury %, to holders %, to dev %, burn %)
- Supply mechanics (fixed, inflationary schedule, burn rate, buyback logic)
- Treasury control (autonomous/PDA-style, multisig, governance-gated, timelock)
- Redistribution rules (how yield/treasury flows back)
- Lock / vesting structures (cliff, linear, milestone-based)
- Governance parameters (proposal threshold, voting power concentration, execution delay)
- Holder incentive mechanisms (staking rewards, loyalty multipliers, etc.)

**Search Strategy:**
- Start with a curated "reasonable bounds" grid (informed by known good/bad examples on Solana)
- Use Latin Hypercube Sampling or Sobol sequences for better coverage than naive grid
- Support both continuous (fee rates) and categorical/discrete (treasury type, vesting shape) variables
- Allow conditional parameters (e.g., if treasury is autonomous, then certain redistribution rules become available)

**Initial Scope (MVP):**
Focus on the most common and impactful primitives first:
1. Fee routing + treasury allocation
2. Supply schedule + burn/buyback mechanics
3. Vesting / lockup design
4. Governance thresholds (to resist capture)

---

### Stage 2: Walk-Forward Validation on On-Chain History

Instead of price candles, use **rolling windows of on-chain token activity**.

**Data Requirements:**
- Historical token launch data on Solana (pump.fun graduates + established tokens)
- Transfer event graphs (who sent to whom, amounts, timing)
- Supply change events (mints, burns, unlocks)
- Holder concentration metrics over time (top 10/100 holders %)
- Volume and liquidity depth
- Treasury flow events (where fees actually went)

**Walk-Forward Setup (adapt the 9-fold approach):**
- Expanding or rolling windows (e.g., 30–90 day training windows)
- Out-of-sample periods that include different market regimes (bull, bear, sideways, high-vol)
- For each token/design candidate, simulate how the design would have behaved if it had been live during that historical window

**Key Metrics to Track per Window:**
- Net value flow to holders vs extraction (fees captured by treasury vs leaked to insiders/extractors)
- Holder retention / concentration changes
- Survival rate (did the token die or keep activity?)
- Resistance to observed attack patterns in history (e.g., large holder dumps, governance proposals that drained treasury)

---

### Stage 3: Darwinian Evolution

Keep the evolutionary layer:
- Population of design candidates
- Multiple generations with selection, crossover, mutation
- Fitness function that rewards designs that perform well across multiple historical windows and stress scenarios

This is where Night Shift's existing Darwinian code can be heavily reused.

---

### Stage 4: Overfitting & Robustness Gates (Critical)

Define stricter gates than the original because tokenomics has higher narrative overfitting risk:

**Proposed Gates (minimum 4):**
1. **IS vs OOS Gap** — Performance degradation from in-sample to out-of-sample must stay below threshold
2. **Cross-Regime Consistency** — Design must not only work in bull markets; test across at least 3 distinct market regimes
3. **Fragility / Sensitivity** — Small changes in parameters or starting conditions must not cause large outcome swings (use Monte Carlo perturbations)
4. **Attack Surface Score** — Explicit simulation of common attack patterns (large holder dump, governance capture attempt, treasury drain proposal, etc.). Designs that fail these are heavily penalized or rejected.

Only designs that pass all gates proceed to full simulation.

---

### Stage 5: Full Agent-Based Economic Simulation

This is the biggest new component compared to the trading version.

**Goal:** Simulate how a token with a given design would behave under realistic (and adversarial) conditions over months/years.

**Agent Types (minimum):**
- Rational holders (long-term, respond to yield signals)
- Speculative traders / flippers
- Large concentrated holders (potential dumpers or governance attackers)
- Treasury / protocol agent (executes redistribution rules autonomously where applicable)
- Attacker agents (try to exploit governance, oracle, or economic parameters)

**Scenarios to Run:**
- Baseline (normal market conditions)
- High sell pressure event
- Governance attack attempt
- Treasury yield generation (if autonomous treasury is part of design)
- Black swan liquidity crunch
- Regulatory / narrative shock

**Outputs per Simulation:**
- Holder value accrual curves
- Treasury growth / depletion
- Concentration metrics over time
- Probability of death spiral or sustained activity
- Resilience Score components

---

## 4. Resilience Score (Core Output)

The Resilience Score should be a **multi-factor, transparent, reproducible score** (0–100 or 0–10 scale) that combines:
- Value Accrual to Holders (fees captured vs extracted)
- Holder Retention & Distribution Health
- Attack Resistance (governance, economic, concentration attacks)
- Sustainability under Stress (multiple regimes)
- Transparency / Verifiability of rules (on-chain enforceability)
- Penalty for hidden extractive mechanisms or high centralization

The score should come with **explainability** — which primitives helped or hurt the score.

This becomes the public methodology for the Observatory.

---

## 5. Tech Stack & Reuse Strategy

**Recommended (to move fast):**
- **Python** (keep the existing Night Shift core where possible)
  - Use or extend the existing grid search, walk-forward, Darwinian, and Monte Carlo modules
- **Data Layer:** Helius / QuickNode / Triton + custom indexer for token events (or start with Dune + Solscan APIs for MVP)
- **Simulation:** Mesa (agent-based modeling framework) or a custom lightweight agent engine
- **Storage:** PostgreSQL or DuckDB for historical windows + results
- **Later (production):** Rust for any hot simulation paths if performance becomes an issue

**What to Clone vs Rewrite:**
- Clone: Grid search engine, walk-forward orchestration, Darwinian evolution loop, Monte Carlo perturbation system, overfitting gate framework
- Rewrite/Adapt heavily: Data ingestion, parameter space definition, fitness function, agent-based simulator, output taxonomy + Resilience Score generator

---

## 6. Phased Delivery (Aligned with cyber•Fund 12-week timeline)

**Phase 1 (Weeks 1–3) — Foundation**
- Define formal tokenomic primitive taxonomy + parameter encoding
- Build data ingestion pipeline for Solana token history
- Implement basic grid search + single-window validation
- Manual curation of 50–100 known tokens as seed dataset (label resilient vs fragile)

**Phase 2 (Weeks 4–7) — Core Engine**
- Implement full walk-forward + Darwinian layers
- Build initial agent-based simulator (basic agents + 2–3 stress scenarios)
- Define and implement first version of Resilience Score
- Run large-scale classification on Solana token set

**Phase 3 (Weeks 8–12) — Observatory Outputs + Hardening**
- Public dataset + basic web explorer
- Annotated patterns + anti-patterns
- Builder Kit examples (how to configure RTP-style treasury for different design types)
- Research notes / methodology paper
- Night Shift Security track scoping (parallel engine)

---

## 7. Risks & Mitigations

- **Data quality / coverage gaps** — Be explicit about what can vs cannot be inferred from on-chain data. Flag low-confidence classifications.
- **Narrative overfitting** — The gates + multi-regime testing + attack simulation are the main defenses.
- **Computational cost** — Start with focused primitive subsets. Use cloud spot instances or Railway for overnight runs.
- **Scope creep** — Strict phase gating. MVP = fee routing + supply mechanics + basic governance. Add complexity only after core loop works.

---

## 8. Success Criteria (for this project)

By end of 12 weeks:
- Night Shift Tokenomics has classified a meaningful set of Solana tokens with reproducible Resilience Scores
- Public taxonomy + dataset exists and is usable by other builders
- At least one validated "resilient pattern" recommendation exists that can be turned into Builder Kit config
- The engine is documented well enough that a future contributor (or Security track) can extend it

---

## Current Implementation Status (as of 2026-06-08)

**Pipeline:** End-to-end working (backfill → classify → Observatory export → regime gate).

**v1 (internal bootstrap baseline):** 51/51 tokens classified, 0 survivors under `min_profitable_regimes=3`. Expected and correct on mostly-synthetic data. Not for publication.

**v2 (current focus):** Partial real backfill complete.
- 6 tokens successfully fetched via Helius (JUP, RAY, ORCA, JTO, JITOSOL, RENDER) with full 270-day histories cached.
- 45 tokens remain on bootstrap synthetic fallback due to Helius credit exhaustion during heavy pagination (JUP alone required 330+ pages).
- Hygiene improvements (120s timeout, exponential backoff retries, pagination logging, manifest provenance) are in place in `src/night_shift/data/ingestion/helius.py`.
- `backfill_manifest.json` already tracks per-token `source`, `bars`, `synthetic_fallback`, and `attempts`.

**Blocker:** Helius quota on high-volume tokens. First public Observatory dataset requires dominant real on-chain provenance.

**Immediate runway:** 15-day QuickNode trial active. This is the bounded window to complete a credible v2 with real data for the remaining 45 tokens.

**Key files (explored on main):**
- `src/night_shift/data/ingestion/helius.py` — `fetch_token_daily_history(mint, days, ...)` → list of `{"timestamp", "amount", "type"}` events. Uses Helius enriched `/addresses/{}/transactions` endpoint + `extract_transfer_events`.
- `src/night_shift/data/loader.py` (and `_load_single_token`) — integration point that calls ingestion, converts to bars, tags `df.attrs["source"]`.
- `src/night_shift/batch/backfill.py` — orchestrates per-token cache check → fetch → manifest update. Never imports helius directly.
- `scripts/run_classification_batch.py` — phase driver (`--phase backfill|classify|all`).
- `data/tokens/backfill_manifest.json` + parquet cache in `data/tokens/cache/`.
- Export: `data/exports/v2_<date>/`.

The original Night Shift research engine patterns (parallel exploration + multi-gate validation) have already been adapted into this modular `src/night_shift/` structure. The immediate task is hardening the data ingestion layer for real scale and provider resilience.

## Data Ingestion Layer – Provider Abstraction (Scoped Increment for v2)

**Goal:** Make the ingestion layer support multiple providers while preserving the exact contract expected by `loader.py` and `run_backfill_phase`. This enables the 45-token real backfill inside the 15-day QuickNode trial and creates a reusable component for both Tokenomics and Security tracks.

**Current implementation (helius.py):**
- Stdlib `urllib.request` only.
- Public function: `fetch_token_daily_history(mint: str, days: int = 180, api_key=None, max_pages=None) -> List[Dict]`
- Returns events in shape `{"timestamp": int, "amount": float, "type": "transfer"|"mint"|"burn"}`.
- Pagination via `before-signature` on Helius enriched tx endpoint.
- Already contains solid timeout (120s), 5 retries with exponential backoff, and per-page logging.
- Limitation: tightly coupled to Helius response shape (`tokenTransfers`, `accountData.tokenBalanceChanges`).

**Required change:**
Add `src/night_shift/data/ingestion/quicknode.py` that exports a function with the **identical signature and return shape**:
```python
def fetch_token_daily_history(
    mint: str,
    days: int = 180,
    api_key: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ...
```

Inside `quicknode.py`:
- Use QuickNode RPC URL (from env var `QUICKNODE_RPC_URL` or equivalent).
- Implement signature pagination (`getSignaturesForAddress` with `before` cursor) + transaction fetching.
- Provide equivalent transfer/mint/burn event extraction (lightweight SPL Token program instruction parsing is sufficient for regime volume/activity signals).
- Reuse the same logging, cutoff, and retry skeleton from helius.py for consistency.
- Tag source as `"quicknode"` in the returned DataFrame attrs (via loader).

**Integration point (minimal change):**
Update the loader (`src/night_shift/data/loader.py` or `_load_single_token`) with a thin provider selector:
- Prefer QuickNode when `QUICKNODE_RPC_URL` (or `DATA_INGESTION_PROVIDER=quicknode`) is set.
- Fall back to Helius or bootstrap as before.
- Preserve all manifest fields (`source`, `synthetic_fallback`, coverage notes) and `df.attrs["source"]` contract.

**Non-goals for this increment:**
- Perfect 1:1 parity on every edge-case inner instruction or Token-2022 variant (document coverage gaps in manifest for affected tokens).
- Full solders-based decoder (stdlib urllib + manual parsing is acceptable for v2 velocity; can be hardened later).
- Multi-provider unit tests or CLI flags (env var is sufficient).

**Success criteria for v2 inside 15-day window:**
1. `quicknode.py` implemented and passes single-token test (e.g. a thin token + one high-volume token).
2. Targeted backfill of the 45 bootstrap-only tokens completes with `source=quicknode` dominant in manifest.
3. `run_classify_phase` runs cleanly with identical gate config.
4. `data/exports/v2_2026-06-xx/` produced with clear "Data Sources & Provenance" section documenting the 6 Helius + 45 QuickNode mix and any coverage notes.
5. 0 changes required to `run_backfill_phase`, classify gates, or export schema.

**Why this matters (long-term):**
A provider-agnostic ingestion layer is high-leverage public infrastructure. It de-risks future Helius quota events, enables Security track historical replay/adversarial precondition mining, and supports eventual multi-chain expansion without rewriting the research engine core.

## Current Sprint Execution Plan (15-day QuickNode Trial)

1. Apply this SPEC update and commit (today).
2. Local coding agent implements `quicknode.py` + minimal loader routing (1–2 days).
3. Test fetch on 2–3 tokens; validate event shape and manifest tagging match Helius path.
4. Targeted backfill run for the 45 remaining tokens only (skip already-cached Helius tokens).
5. Classify + v2 export with provenance documentation.
6. Review regime distribution vs v1; decide on publication or quick sensitivity pass.
7. Push resulting v2 artifact + updated manifest/spec as the first credible public Observatory dataset.

**Assumptions:**
- QuickNode trial provides sufficient archive RPC throughput and historical depth for 270-day windows on the remaining tokens.
- Minor differences in parsed transfer volume between providers will be visible in manifest coverage stats and will not invalidate the overall v2 methodology.
- Helius startup program decision may land during or after this window; a homogeneous v3 run can be scheduled later if needed.

**Risks (documented):**
- Raw transaction parsing in quicknode.py may have lower recall on complex cases → mitigated by manifest notes and transparent export section.
- Trial deadline is hard → scope is deliberately limited to shipping one real-data-primary v2 export.

This increment directly extends the existing modular design (`data/ingestion/`, loader abstraction, manifest provenance) rather than introducing new architecture. It preserves the core Night Shift philosophy of brutal validation while removing the immediate data-source blocker.

**Next action for local coding agent:** Implement `quicknode.py` per the interface and success criteria above. All other files remain unchanged for this increment.

---

This spec is designed to be fed directly to an AI coding agent (Claude, Cursor, etc.) or a developer. It preserves the battle-tested strengths of the original Night Shift while making the necessary domain shift to tokenomics.

---

**Next:** Night Shift Security edition spec (parallel but distinct research track).