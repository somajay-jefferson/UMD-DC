# CLUSSO Paper — Understanding Summary

**Rubin, Fan, Barisoni, Janowczyk, Zee. "Novel Scalar-on-matrix Regression for Unbalanced Feature Matrices." Statistics in Biosciences (2026) 18:192–213.**

## 1. Problem being solved

Digital pathology can segment individual objects (e.g. tubules) from a whole slide
image and compute a fixed set of `q` image features for each object. So each subject
`i` is represented not by a feature *vector* but by a feature *matrix*
`X_i ∈ R^(p_i × q)`, where `p_i` (number of tubules) varies freely across subjects
and `q` (number of features) is fixed. Goal: regress a scalar clinical outcome
`y_i` (e.g. eGFR, kidney function) on `X_i` and — critically — recover *which of the
q features* actually drive the outcome (variable selection / biomarker discovery),
not just predict well.

**Why existing methods don't work:**
- Vectorizing `X_i` loses row/column structure and blows up the parameter count.
- Existing scalar-on-matrix regression (trace regression [Zhou & Li], structured
  lasso [Zhao & Leng]) all require a **balanced design** — every subject's matrix
  must have the same number of rows. Real tubule counts per subject vary (in
  their data: median ~900–930 tubules/subject, IQR roughly 570–1400) so this
  doesn't hold.
- Naively averaging every tubule's features into one vector per subject (the
  "naïve" baseline throughout the paper) throws away within-subject heterogeneity
  — e.g. atrophic vs. non-atrophic tubules have different tubular basement
  membrane thickness, so averaging over both washes out a real bimodal signal.

## 2. Full Information Structured Lasso (Section 2) — the oracle

Assume (hypothetically) each subject's *entire population* of tubules is known and
correctly labeled into `G = 2` latent subgroups. Then define:
- `X*_i ∈ R^(2×q)`: row `k` = the subject's population-average feature vector for
  subgroup `k`.
- `w_i^k`: true proportion of subject `i`'s tubules in subgroup `k`.
- `X**_i = diag(w_i^1, w_i^2) · X*_i` — the weighted, balanced 2×q design matrix.

Model: `y_i = (α*)ᵀ X**_i β* + ε_i`, fit by adapting Zhao & Leng's structured lasso:

```
(α̂*_F, β̂*_F) = argmin_{α*,β*} (1/n) Σ (y_i − α*ᵀ X**_i β*)² + λ_n ‖β*‖_1
  subject to: ‖α*‖_1 = 1, sign(β*_(1)) = 1   (identifiability constraints)
```

`α*` (row/cluster-level effect) is **not** penalized — both clusters are always
used. `β*` (column/feature-level effect) **is** L1-penalized — this is where
sparse feature selection happens. This estimator is called "Full Information"
because it assumes perfect knowledge of cluster membership and the full tubule
population — it's an oracle, not usable on real data, but it has a proven L1
error bound (Eq. 3-4, inherited from Zhao & Leng's restricted-eigenvalue
condition) and serves as the ceiling that CLUSSO is compared against.

## 3. CLUSSO (Section 3) — the actual estimator

In reality: cluster labels are latent (unknown), and only a *sample* of `p_i`
tubules per subject is observed, not the full population. CLUSSO's fix:

1. Pool **all** tubules from **all** subjects into one big matrix `𝕋`.
2. Run unsupervised clustering (paper uses GMM/EM as default; also tests K-means)
   on `𝕋` to assign every tubule to one of `G = 2` clusters. This is a **joint**
   clustering step across the whole cohort, not per-subject.
3. For each subject, average feature values within each estimated cluster to get
   `X̂*_i ∈ R^(2×q)` (Eq. 5). A subject needs ≥1 tubule in each cluster or their
   row is undefined (i.e. they get dropped — this matters for real data).
4. Estimate `ŵ_i^k` = observed sample proportion of that subject's tubules in
   cluster `k`. Weight: `X̂**_i = diag(ŵ_i^1, ŵ_i^2) X̂*_i`.
5. Fit the same structured-lasso objective as the oracle, but on `X̂**_i`
   (Eq. 6) — this is the CLUSSO estimator `(α̂*_C, β̂*_C)`.

**Key theoretical result (Section 3.2):** under *perfect* clustering
(`𝕋̂_k = 𝕋_k`), CLUSSO is shown via the Weak Law of Large Numbers + continuous
mapping theorem to converge in distribution to the Full Information Structured
Lasso as the per-subject tubule counts →∞. So CLUSSO is asymptotically justified
as an approximation to the (unobservable) oracle, with the approximation quality
governed by clustering accuracy and per-subject sample size.

This is exactly the alternating-optimization machinery in this repo's
`Mainfunction_albet.py`: fix `α`, solve lasso for `β`; fix `β`, solve OLS for
`α`; repeat to convergence. `coefficient.py` is the standalone lasso-path/BIC
fitter used for the naïve baseline; `K_prdu.py`/`mat_vec_prd.py` are the tensor
helpers for that alternation.

## 4. Simulation study (Section 4) — design

- `n ∈ {300,...,2000}` subjects, `q ∈ {10,...,200}` features, sparsity
  `s_β* ∈ {0.4, 0.8}` (proportion of zero β* entries), `G = 2` fixed.
- Cluster 1 features ~ N(2,1), cluster 2 ~ N(5,3), per-subject cluster weight
  `w_i^1` drawn from {0.2,...,0.8}.
- Tubule count `p_i` ~ discrete uniform with mean 40, variance 5 (giving range
  ~37-43) — this maps directly to `mu`/`sigma_sq_m` in this repo's
  `CLUSSO_Functions_Project1_6_16_23.py`.
- Observed tubules generated by resampling from the 2 cluster means with
  measurement noise `σ_R² = 1` by default; robustness checks vary `σ_R²` up to
  46, add tubule correlation (matrix-normal generation), misspecify the number
  of true clusters (1 or 3 instead of 2), and swap GMM for K-means clustering.
- Compared: CLUSSO vs. naïve (plain lasso on per-subject tubule averages) vs.
  Full Information Structured Lasso oracle. Metrics: TPR, FPR, L1 bias of β̂*,
  prediction MSE, clustering accuracy. 1000 reps/setting, median reported
  (except clustering accuracy, which uses means).
- λ selected via 5-fold CV over a fixed grid, coefficients below 0.001
  (after L1-normalization) hard-thresholded to zero — same convention used in
  this repo's Python ports (`beta_thresh = 0.001` in `CLUSSO_Data_Example.py`).

## 5. Simulation results — what actually happens

- **TPR:** Full Information > CLUSSO ≳ naïve, for all settings. All non-decreasing in `n`.
- **FPR:** this is where CLUSSO wins clearly. Naïve's FPR **increases** with `n`
  (converges toward OLS as λ shrinks — it just keeps adding predictors).
  CLUSSO's FPR generally **decreases** with `n`. At high sparsity (`s_β*=0.8`),
  CLUSSO needs a smaller `n` than naïve to beat it on FPR.
- **Bias:** CLUSSO < naïve, both > Full Information, across all settings.
- **MSE:** CLUSSO ≈ naïve (comparable), both > Full Information oracle.
- **Robustness:** CLUSSO's FPR advantage over naïve *persists* as `σ_R²`
  increases (i.e. even as GMM clustering accuracy degrades from ~100% down to
  ~74%), when tubules are correlated, and under 3-true-cluster misspecification
  (forced into 2 estimated clusters). CLUSSO is also insensitive to GMM vs.
  K-means choice, despite the two algorithms only agreeing on ~87% of individual
  tubule labels in one high-noise check.
- One-cluster case (no true subgroup structure) isn't reported: clustering
  collapses everything into one group and most subjects get dropped for lacking
  a tubule in both clusters — a real practical failure mode.

## 6. Real data application (Section 5)

- NEPTUNE (n=250, training) and CureGN (n=261, external validation) cohorts,
  glomerular disease. Outcome: eGFR at biopsy (CKD-EPI/U25 formulas). 57
  hand-crafted tubular image features (area, thickness, smoothness, various
  ratios) from a pretrained deep-learning tubule segmentation model.
- `G=2`, λ grid {0.1,...,20}, chosen λ=19.7 via 5-fold CV; β̂* selected as the
  minimizer of the CLUSSO objective across 5 random initializations of α*, β*
  (structured lasso only converges to a stationary point, not global minimum —
  this is why `CLUSSO_Data_Example.py` in this repo runs `M=5` random restarts
  and keeps the best by objective value).
- **CLUSSO selected exactly one feature**: "tubular epithelium + lumen area."
  Naïve selected three different features (lumen avg thickness, TBM smoothness,
  TBM/tubule diameter ratio) — no overlap with CLUSSO's pick.
- MSE on NEPTUNE (train): naïve 779 vs. CLUSSO 757 (3% reduction).
- MSE on CureGN (external validation): naïve 1177 vs. CLUSSO 1094 (7%
  reduction) — using whichever of the two possible train/test cluster-label
  correspondences gave the lower MSE (there's a label-switching problem: cluster
  "1" in the test set clustering isn't guaranteed to correspond to cluster "1"
  in training, so both correspondences must be tried and the better one kept —
  picking the wrong one gave MSE 6205, dramatically worse).
- Interpretation: CLUSSO's single selected feature is more parsimonious, more
  generalizable out-of-cohort, and biologically consistent with tubular atrophy
  (atrophic tubules are known to show tubular lumen contraction / flattened
  epithelium, i.e. reduced epithelium+lumen area).

## 7. Explicitly stated future directions (Section 6) — relevant to your idea

The paper's own discussion lists these extensions:
- Binary outcome → matrix-variate logistic regression (Hung & Wang).
- Survival outcome → L1-penalized Cox model for matrix predictors.
- Adding subject-level scalar covariates directly into the SSE term.
- Swapping plain lasso for **random lasso** to better handle correlated features.
- **"The work of He et al. [9] could be employed to control the false discovery
  rate of CLUSSO at a pre-specified threshold."** — this is a direct citation of
  the second paper (`False_Discovery_Rate_Control.pdf`), explicitly flagged as
  unfinished business by CLUSSO's own authors.
- Extending Ou-Yang et al.'s sparse tensor regression to generalize CLUSSO from
  2D unbalanced predictors to unbalanced arrays of >2 dimensions — this is
  essentially what this repo's `src/tepig/` code already does (adding a "slide"
  mode `S` on top of cluster `G` and feature `q`).
- Allowing different subjects to have different numbers of latent clusters.
- Studying CLUSSO's performance under different measurement-error models for
  tubule misclassification.

## 8. Key limitations to keep in mind

- Signs of β̂* are not identifiable (only α*·β* product is estimated), so only
  coefficient *magnitudes* can be compared/ranked, not direction of effect.
- Cluster-label correspondence between train and test sets is not automatic and
  must be resolved by trying both permutations and picking by validation MSE —
  this gets combinatorially worse if `G > 2`.
- No built-in false discovery control: λ is chosen purely by CV-minimized MSE,
  which (per the companion FDR paper) is known to not control the false
  discovery rate of the resulting selected-feature set.
- Subjects with zero tubules in some cluster are silently dropped from the
  clustering-accuracy calculation and can cause resampling loops in the
  simulation code (`no_clust` retry logic in `CLUSSO_Functions_Project1_6_16_23.py`).
