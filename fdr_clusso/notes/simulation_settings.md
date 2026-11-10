# Simulation settings side by side

Our current headline run (`stage6_exactk_fine_reps100.json`) next to the three
sources we could borrow from. Compiled 2026-10-25, updated 2026-10-27.

| parameter | **this run** | CLUSSO paper | Rubin's R code | PS-Fdr paper (He et al.) |
|---|---|---|---|---|
| replications | **100** (mean, median, 95% CI) | 1000 per setting (medians) | 1000 (CLUSSO sims) | 100 per setting (means) |
| subjects n | **300** | 300, 500, …, 2000 | 100 (Random CLUSSO); about 511 (real-data sim) | 550, 1000 (n>p); 500 (n<p) |
| features | **50** | 10, 50, 100, 150, 200 | 10; about 46 (real tubule features) | 500; 1000 |
| truly real features | **10 of 50 (20%)** | 60% or 20% | 40% of 10; 3 of 46 | 20 of 500 (4%); 10–50 of 1000 (Model F) |
| effect sizes | **β, α drawn from {1,2,3,4}, all positive** | same | β = 3; β = 15 (real-data sim) | 0.25–0.40 (A/B), 0.2–0.5 (C), U(0.2, 1) (F); **random signs** |
| feature correlation | **none** | none (one check correlates *tubules* at 0.5) | **ρ = 0.9 among real features** | **AR(1), ρ = 0.6, blocks of 100** |
| tubules per subject | **37–43** (mean 40, var 5) | same | mean 100 | n/a (no tubules) |
| cluster means / sds | **N(2, 1), N(5, 3)** | same (+ N(7, 3) in 3-cluster check) | N(5, 1), N(10, 3); real tubules | n/a |
| tubule measurement noise σ_R² | **1** | 1; robustness 1–46 | 1 | n/a |
| outcome noise σ² | **1** | 1 | 1 | standard normal |
| clustering | **GMM, 2 clusters, once per dataset (frozen)** | GMM; K-means check | GMM (`Mclust`) | n/a |
| B / M | **50 / 100** | — | B = 200 or 5 (Random CLUSSO) | 50 / 100 |
| λ grid | **0.01–20, 40 log steps, both arms** | 0.5–5 by 0.5, 5-fold CV | 0.1–20 by 0.1; 0.1–5 (real-data sim) | none (exact lasso path) |
| starts per fit | **5 fixed** | random | random | n/a |
| coefficient threshold | **0.001** | 0.001 | **0.01** | n/a |
| FDR target q | **0.10** | n/a | n/a | 0.10 |
| methods compared | **naive, CLUSSO alone, FDR-CLUSSO, oracle** | CLUSSO, naive, Full Information | + Random CLUSSO | PS-Fdr, CV-lasso, knockoff, univariate BH |
| metrics | **FD, FN, FDP, power, FD+FN, TPR, FPR, L1 bias, MSE, time** | TPR, FPR, L1 bias, MSE, clustering accuracy | same as CLUSSO paper | FD, FN, FDP, power, FD+FN |

## Where each column comes from

- **This run:** `stage6_exactk_fine_reps100.json` (`args`, `meta`), `stage1_noise.py` (`FIXED`, `make_cohort`),
  `clusso_select.py` (`DEFAULT_LAMBDA_GRID`, `CORE_BETA_THRESH`).
- **CLUSSO paper:** `CLUSSO_Paper.pdf`, Section 4.1 (data generation) and 4.2.
- **Rubin's R code:** `clussocode.zip` → `CLUSSO_Simulations_Project1_6_16_23.R`;
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
- Stage 4's 20 reps (FDP 0.138) couldn't separate 0.138 from 0.10. The
  100-rep run settles it: 0.113 [0.093, 0.132]. See `RESULTS.md`.
