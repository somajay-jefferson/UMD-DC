# Grid figures in the Random CLUSSO paper's layout: what we have, what to change

**Goal:** figures like Figs 1–5 of Rubin et al. 2026, *Statistical Analysis and
Data Mining* (`Random_CLUSSO.pdf`, pages 6–9). One figure per metric (TPR, FPR,
bias, MSE, time), a panel per (ρ, q), x = n, one line per method, medians over
replications. We add FDP, since FDR is what FDR-CLUSSO is for.

**The code to do this now exists:**

- `run_grid.py` runs `stage4_clusso_fdr.py` once per (n, q) cell. Each output is
  named for exactly what ran and carries its `meta` block. Cells that are
  already done are skipped, so the grid can fill in over days.
- `plot_grid.py` draws the figures from whatever cells exist. Missing cells show
  as "not run", a lone n is a single point, and two runs of one cell never
  silently mix (the one with more replications wins, and the clash is
  reported).

What follows is the gap between their setup and ours, parameter by parameter.

## Their figure setup vs ours

Status: ✅ = matches now · 🟡 = available through a flag · 🔴 = needs a code
change

| | Random CLUSSO paper (§3) | ours now | status |
|---|---|---|---|
| **n** | 200, 300, 400, 500, 1000 | any (`--n`) | ✅ |
| **q** (features) | 20, 50, 100 | any (`--q-features`) | ✅ |
| **ρ** (feature correlation) | 0, 0.3, 0.6, 0.9 | `--design random_clusso --rho` | 🟡 #1 done |
| latent tubule matrix X* | matrix-normal: means (5, 10), row cov diag(1, 3), column cov with an exchangeable-ρ block on the middle 40% | same, with `--design random_clusso` (default design: independent N(2, 1), N(5, 3)) | 🟡 #1 done |
| **which features are real** | **fixed:** the middle 40% (the correlated block) | same, with `--design random_clusso` (default: random positions, `--sparsity`) | 🟡 #1 done |
| **β on real features** | **3** for all | same, with `--design random_clusso` (default: random from {1, …, 4}) | 🟡 #1 done |
| α | {1, 2, 3, 4} | {1, 2, 3, 4} | ✅ |
| tubules per subject | mean 100, var 5 | mean 100 with `--design random_clusso` (default design: mean 40) | 🟡 #1 done |
| tubule noise σ_R², outcome noise σ² | 1, 1 | 1, 1 | ✅ |
| cluster weights w | {0.2, …, 0.8} | {0.2, …, 0.8} | ✅ |
| clustering | GMM, 2 clusters | GMM, 2 clusters, frozen once per dataset | ✅ |
| **λ grid** | {0.1, …, 1, 1.5, …, 5}, 5-fold CV | 0.01–20, 40 log steps, 5-fold CV | 🔴 #2 |
| **coefficient threshold** | **0.01** | **0.01** (`--thresh`, default since #16; core keeps 0.001 for R parity) | ✅ #3 done |
| starts per fit | one random start | 5 fixed starts | keep ours (REVIEW §1: random starts make the selection rule non-deterministic) |
| **replications** | **250** | any (`--reps`); 100 so far | 🟡 cost, see below |
| summary | median (time: mean) | `plot_grid.py --stat median` (default) | ✅ |
| TPR, FPR, L1 bias, time | yes | yes, every method | ✅ |
| MSE in the figures | **in-sample**: train on all subjects, predict all | in-sample by default | ✅ (keep `--holdout 0` for these figures) |
| held-out MSE | real data only: 80/20 split, both cluster-label orders | `--holdout 0.2`, same protocol; refit trained on the training split only | 🟡 |
| methods | Random CLUSSO, CLUSSO, naive, Full Information | CLUSSO, naive, Full Information, **FDR-CLUSSO** (exact-k, one-λ) | Random CLUSSO 🔴 #4 |
| extra settings | within-cluster mean variation (n=500, q=50, ρ=0.6); real-data-like design (Table S2) | — | 🔴 #1 (first); second needs NEPTUNE/CureGN data |

## What needs to change, in order

**#1. Their data generator, behind a `--design random_clusso` flag (unlocks the ρ columns). ✅ Done:** `random_clusso_design.py`, wired into `stage4_clusso_fdr.py` (`--design`, `--rho`) and `run_grid.py` (`--design`, `--rho` list; default `random_clusso`). Checks: the correlated block correlates at exactly ρ; at ρ = 0 it matches core's generator. The original plan follows.
- This is a port of `generate.data` in `Supplemental_Code.zip/Random_CLUSSO_non_mainfunctions.R`.
- It's smaller than it looks. Their generator is the same as core's except for one draw: each subject's latent 2 × q matrix comes from a matrix normal instead of independent normals.
  - At ρ = 0 the two are the same distribution, just with different constants.
  - So the change is that one draw (X = M + A Z Bᵀ, done in numpy, with no MixMatrix dependency) plus their constants: means 5/10, 100 tubules, β = 3 on the middle 40%.
- It belongs in `src/fdr_clusso/`, not `src/core/`. Core's generator stays untouched, and the R file is its twin.
- New args: `--design`, `--rho`. `run_grid.py` gets a `--rho` list. `plot_grid.py` already places runs by `args.rho`.
- **Effort:** small to medium. **Check needed:** at ρ = 0, the port's summary statistics should match core's generator with the constants swapped.

**#2. A `--lambda-grid paper` option.**
- This threads their 14-value grid into `clusso_fit_cv`, `oracle_fit` and `ClussoSelector`.
- With `--null-ladder fine`, the permuted arm uses the same grid.
- **Effort:** small.
- **Risk:** their grid tops out at λ = 5. Our permuted arm needs a λ sparse enough to land under k, so check `exhausted_rate` on a quick run before trusting it.

**#3. The 0.01 coefficient threshold. ✅ Done in #16:** `--thresh`, default 0.01 for the whole pipeline; core keeps 0.001 as its own default. The original reasoning follows.
- 0.001 is hardcoded inside core's `Mainfunction_albet`. Matching theirs means a parameter in `src/core/`, which is REVIEW open question 3. Default 0.001 would change no existing result.
- It matters. The stage 4 and 5 runs found the cut plausibly deciding bare CLUSSO's support on 44–60% of cohorts (REVIEW §6 has the mechanism), so 0.01 will change TPR/FPR, especially for CLUSSO.
- **Decision needed:** edit core, or plot with ours and say so on the figures.

**#4. Random CLUSSO itself, as the fourth line (optional).**
- This means porting the two-round bootstrap from `Random_CLUSSO_Functions_Simulate_Real_Data.R`:
  - B = 200 per round
  - √q features subsampled per bootstrap, weighted by first-round importance
  - CV-tuned CLUSSO fit per bootstrap
  - structured-OLS refit (already ported)
- **Effort:** large. **Cost:** about 400 CV-tuned CLUSSO fits per replication. Their paper reports up to about an hour per replication on an 80-core node.
- Without it, our figures compare FDR-CLUSSO against CLUSSO, naive and Full Information, which is three of their four lines.

## Cost

Measured: one cell (n=300, q=50) at 100 replications took 22 h with the
paper's rule (40-step ladder). One-λ is about 10x cheaper. The cost grows with
n and q; the estimates below scale it roughly by n × q and are not measured.

| grid (one-λ arm, 50 reps, in-sample) | cells | estimate |
|---|---|---|
| n × q, ρ = 0 (works today, our generator) | 15 | ~1–1.5 days |
| n × q × ρ (after #1) | 60 | ~4–5 days |
| paper's rule instead of one-λ | — | ~7–10x the above |
| 250 reps instead of 50 | — | 5x the above |

**Suggested order:** #1 → a quick test grid → decide #3 → run n × q at ρ = 0 and
ρ = 0.6 first (the column where the paper's methods separate) → fill the rest.
Use one-λ to scan, then spot-check 2–3 cells with the paper's rule.
