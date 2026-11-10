# False Discovery Rate Control Paper — Understanding Summary

**He, Zhou, Jiang, Wen, Li. "False discovery control for penalized variable selections with high-dimensional covariates." Stat Appl Genet Mol Biol 2018;17(6).**

## 1. Problem being solved

Penalized regression (lasso, elastic net, etc.) is the standard way to do variable
selection when `p >> n` (e.g. tens of thousands of genes, hundreds of subjects).
The regularization parameter `λ` is almost always chosen by K-fold
cross-validation to minimize prediction error. **This paper's core empirical
claim: CV-selected λ does not control the false discovery rate (FDR) of the
selected variable set — it systematically selects far too many irrelevant
variables.**

Demonstration (Section 2.2): 340 patients, 23,052 real gene expression
predictors, outcome simulated so only 50 predictors truly matter. Running lasso
+ 10-fold CV 10,000 times gives a false-discovery-proportion histogram
concentrated around **0.90** (Fig. 1a) — i.e. ~90% of the "significant" genes
CV-tuned lasso reports are actually noise. This is the direct analogue of what
this repo's CLUSSO paper found about the naïve lasso baseline's FPR inflating
with `n` — same underlying mechanism (CV-λ trends toward the unregularized/OLS
solution as effective sample size grows, admitting more and more noise
variables).

## 2. Why standard fixes don't apply here

- **Classical FDR control (Benjamini-Hochberg)** needs valid p-values per
  variable. Penalized estimators in high dimensions don't have a known/tractable
  limiting distribution, so valid p-values generally aren't available.
- **Permutation testing** is invalid here too: under a straight permutation of
  the outcome, *no* variable is truly associated with `y` — but in the real
  (non-permuted) data, some variables *are* truly associated. Because
  penalized selection is a joint, competitive process (variables compete for a
  finite "selection budget" under the L1 constraint), the presence of real
  signal in the original data changes how easily the *noise* variables get
  selected, relative to a fully-null permuted dataset. So permutation
  systematically **under-estimates** the true number of false discoveries,
  and any FDR procedure built directly on that comparison is invalid (cites
  Barber & Candès 2015, Section 3.1).
- **Knockoff filters** (Barber & Candès 2015) do work in principle but are
  designed for `n > p` and lose substantial power in smaller-`n`/high-noise
  regimes — shown empirically in this paper's own simulations.

## 3. The proposed method: PS-Fdr

The key trick is to not compare original vs. permuted data on selection status
directly (which is invalid, see above), but instead to compare **selection
frequency under bootstrap resampling**, and use permutation only to calibrate
the *null distribution of that frequency statistic* — not to directly infer
which specific variables are null.

**Step 1 — Stability selection (Meinshausen & Bühlmann 2010).**
Bootstrap the original data `B` times (default B=50). For each bootstrap
resample, fit lasso and record the selected-variable index set `S^(b)`. The
**selection frequency** for variable `j` is:
```
Π_j = (1/B) Σ_b 1(j ∈ S^(b))
```
This ranks variables by how *robustly* they get selected across resamples,
rather than by a single λ-tuned fit. Note this is largely insensitive to the
exact regularization parameter chosen (Remark 3) — relative rankings are stable
even if the magnitude of Π_j shifts with λ.

**Step 2 — Permutation-calibrated null.**
Permute the outcome `y` (decoupling it from all covariates) `M` times (default
M=100). On each permuted dataset, redo stability selection to get `Π̃_j^(m)`.
Critically: on permuted data, **do not** use CV to pick the number of selected
variables (this would again under-estimate falses) — instead fix the number
of selected variables to match the median count selected on the real data.
Average across permutations: `Π̄_(j) = (1/M) Σ_m Π̃_(j)^(m)` (computed on
*ordered* statistics, so this is really a null distribution over ranks, not
matched by literal variable identity).

**Step 3 — SAM-style normalization.**
To make the "real" and "permuted-null" selection-frequency statistics
comparable, apply the Significance Analysis of Microarrays (Tusher et al. 2001)
normalization: for `u ∈ [0,1]`, `D(u) = u / (sqrt(u(1−u)) + ν)` with small
`ν` (e.g. `1/B`) to avoid division by zero. Apply `D(·)` to both the real
statistic `Π_(j)` and its permuted counterpart to get `Z_(j)` and `Z̄_(j)`.

**Step 4 — Step-down cutoff + Fdr estimate (Algorithm, Section 2.4):**
1. For a threshold gap `Δ`, find the smallest ordered `Z` value such that
   `Z_(j) ≥ Z̄_(j) + Δ` — a step-down search over the ordered statistics.
2. Count `N₊(Δ)` = number of real Z-values at/above this cutoff (candidate
   selected set size).
3. Estimate expected false discoveries `ê₀(Δ)` = average count of *permuted*
   Z-values that would also clear this same cutoff.
4. `F̂dr(Δ) = ê₀(Δ) / N₊(Δ)`.
5. Sweep `Δ` over all observed gaps; for a user-chosen target level `q`
   (e.g. 0.1), select the largest variable set whose `F̂dr(Δ) ≤ q`.

This gives a selected index set with a formal guarantee: at most (roughly)
proportion `q` of the selected variables are expected to be false positives —
directly analogous to Benjamini-Hochberg's guarantee, but built for penalized
high-dimensional selection instead of per-variable hypothesis tests.

**Assumption made explicit (Remark 4):** they follow Benjamini-Hochberg's
conservative convention of setting `π₀ = 1` (assume all variables could be
null when estimating `e₀`), so the resulting `F̂dr` is an *upper bound* on the
true FDR, not an exact estimate.

## 4. Simulation results — what actually happens

Compared: PS-Fdr, 10-fold-CV lasso, knockoff (Barber & Candès), and a classical
univariate Benjamini-Hochberg baseline ("Univariate FDR"). Target level `q=0.1`
throughout. B=50 bootstraps, M=100 permutations for PS-Fdr.

- **n > p (Models A, B):** CV-lasso has FDP around **0.74–0.76** — massively
  over-selecting — in every single setting tested, regardless of effect size.
  Knockoff controls FDP well but has visibly lower power, especially at smaller
  n (Model A, n=550): e.g. knockoff power ~0.10–0.15 vs. PS-Fdr power ~0.92–1.0
  at comparable/lower FDP. PS-Fdr keeps FDP near the 0.1 target while achieving
  power comparable to or exceeding knockoff, and much higher than knockoff in
  the harder (smaller n) regime.
- **n < p (Models C–H, covering linear/logistic/Cox survival):** Univariate FDR
  (classical BH on per-variable p-values) here systematically **fails to control
  FDP** — because it ignores correlation/joint structure — running FDP as high
  as 0.4–0.7 depending on setting, while PS-Fdr holds FDP near 0.1 across all of
  them and has *higher* power than Univariate FDR in essentially every setting
  (Figs. 2–3). This holds across linear regression, logistic regression, and Cox
  proportional-hazards models — i.e. the method is not tied to squared-error
  loss.
- **Tuning sensitivity (Fig. 4):** varying the underlying λ used doesn't move
  PS-Fdr's power/FDP much; 50–100 bootstraps is enough for stable estimates;
  performance is robust to the number of permutations M.

## 5. Real data results

- **Multiple myeloma survival data** (gene expression + Cox model, n=340
  train / 214 validation): CV-lasso selects 21 genes; PS-Fdr selects 11 (a
  strict subset) at F̂dr ≤ 0.1. On external validation, PS-Fdr's smaller set
  achieves a **higher** C-index (0.713) than CV-lasso's (0.677) — fewer,
  better-controlled selections generalize better. (A competing method,
  "MS-Split," selected zero variables — too conservative to be useful here.)
- **Type 2 diabetes gene expression data** (n=271, p=21,735, linear model):
  CV selects 235 genes; PS-Fdr selects only 1 gene at F̂dr≤0.1 (3 genes at
  F̂dr≤0.2). Despite selecting vastly fewer variables, 10-fold CV prediction
  error is nearly identical (23.27 vs. 23.92) — most of CV-lasso's 235 "hits"
  were adding noise, not predictive signal.

## 6. Method properties relevant to combining with CLUSSO

- **Model-agnostic by design**: works with any loss function optimized by a
  penalized selection procedure — explicitly demonstrated for linear, logistic,
  and Cox regression. The paper frames it as pluggable onto "a broad class of
  variable selection algorithms," not lasso-specific.
- **Does not require a known limiting distribution** of the selection
  estimator (this is exactly the obstacle CLUSSO would otherwise face, since
  CLUSSO's alternating structured-lasso estimator's finite-sample/limiting
  distribution isn't derived in the CLUSSO paper — only an asymptotic error
  *bound*, not a distribution).
- Its unit of inference is "is variable j selected or not" via a resampling
  frequency — this maps naturally onto CLUSSO's β* feature-selection question
  (which of the q image features are truly nonzero), largely orthogonal to
  CLUSSO's separate α*/clustering machinery.
- Cost: requires `B` bootstrap refits × `M` permutation refits × (however many
  Δ/λ values are swept) — computationally this is `O(B·M)` full model fits,
  which for CLUSSO (already expensive: GMM clustering + alternating structured
  lasso + CV over a λ grid + multiple random restarts) could be a serious
  compute-cost multiplier depending on how the two are combined.

## 7. Explicit link back to CLUSSO

The CLUSSO paper's Discussion section names this exact paper (reference [9],
He et al. 2018) as a concrete, unimplemented extension: *"The work of He et al.
[9] could be employed to control the false discovery rate of CLUSSO at a
pre-specified threshold."* CLUSSO currently controls only λ via CV-minimized
MSE, with a hard 0.001-post-hoc-threshold rule on normalized β̂* — this is
precisely the kind of ad hoc, uncontrolled selection rule this FDR paper shows
(via the 10-fold-CV lasso baseline) tends to produce FDP around 0.75–0.90 in
comparable high-dimensional settings.
