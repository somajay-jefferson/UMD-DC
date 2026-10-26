# Simulation settings side by side

Our stage 4 setup next to the three sources we could borrow from. Compiled
2026-10-25.

| setting | ours (stage 4) | CLUSSO paper | Jeremy's R code | PS-Fdr paper (He et al.) |
|---|---|---|---|---|
| replications | **20** | 1000 (reports medians) | 1000 | **100** (reports means) |
| n | 300 | 300–2000 | 100 (Random CLUSSO), about 511 (real data) | 550, 1000 (n>p) |
| features | 50 | 10–200 | 10, about 46 (real) | 500 |
| truly real features | 10 of 50 (20%) | 60% or 20% | 40% of 10; 3 of 46 | **20 of 500 (4%)** |
| effect size | 1–4, all positive | 1–4, all positive | 3; 15 (real) | **0.25–0.40, random signs** |
| feature correlation | none | none (tubules 0.5 in one check) | **ρ=0.9 among real features** | **blocks of 100, AR1 ρ=0.6** |
| B / M | 50 / 100 | — | B=200, or 5 | 50 / 100 |
| λ grid | 0.01–20, 40 log steps | 0.5–5 by 0.5, 5-fold CV | 0.1–20 by 0.1 | none (exact lasso path) |
| coefficient threshold | 0.001 | 0.001 | **0.01** | — |
| what gets reported | FDP, power, number selected | TPR, FPR, bias, MSE | same | FD, FN, FDP, power, FD+FN |

## Where each column comes from

- **Ours:** `stage4.json` (`args`), `stage1_noise.py` (`FIXED`, `make_cohort`),
  `clusso_select.py` (`DEFAULT_LAMBDA_GRID`, `CORE_BETA_THRESH`).
- **CLUSSO paper:** `CLUSSO_Paper.pdf`, Section 4.1 (data generation) and 4.2.
- **Jeremy's R code:** `clussocode.zip` → `CLUSSO_Simulations_Project1_6_16_23.R`;
  `Supplemental_Code.zip` → `Random_CLUSSO_Data_Analysis.R`;
  `Random_CLUSSO_Simulations_Simulate_Real_Data.R` (needs the NEPTUNE/CureGN
  `.sas7bdat` files, which aren't in the repo).
- **PS-Fdr paper:** `False_Discovery_Rate_Control.pdf`, Section 3 (Models A–H)
  and Tables 1–2.

## Notes

- Our base case (n=300, q=50, sparsity 0.8, μ=40, clusters N(2,1)/N(5,3),
  weights 0.2–0.8, β from {1,…,4}) is one point on the CLUSSO paper's own grid.
  The gaps are all against the PS-Fdr paper.
- The CLUSSO paper's "correlated" check correlates *tubules* (row covariance,
  0.5). The other two correlate *features*. They're different things.
- Stage 4's FDP of 0.138 has sd 0.117 over 20 reps, so its standard error is
  about 0.026. That's only about 1.5 standard errors above the 0.10 target.
