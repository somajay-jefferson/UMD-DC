# CLUSSO / TEPIG

**Documentation:** https://somajay-jefferson.github.io/UMD-DC/ — the methods
drawn end to end, with a worked example computed from the code in this repo.

Statistical methods for regression on subjects whose data consists of many
object-level measurements grouped into latent clusters (e.g. per-subject
tubule-level histology features aggregated into a scalar outcome such as
`eGFRatBx`). Rather than collapsing each subject to a single averaged feature
vector, **CLUSSO** learns a cluster-weighting vector `alpha` and a
feature-sparsity vector `beta` jointly via an alternating structured lasso.
**TEPIG** extends this to a third "slide" mode, fitting a `(cluster, feature,
slide)` tensor per subject instead of a single matrix.

The repo contains a Python implementation (actively developed, includes the
TEPIG extension and CV-based simulation studies) and the original R
implementation it was ported from.

## Directory structure

```
docs/       GitHub Pages site (served from master /docs)
src/
  core/     Core CLUSSO: alternating structured lasso on (cluster, feature) matrices
  tepig/    TEPIG extension: tensor model on (cluster, feature, slide) data
Random_CLUSSO_*.R      "Random CLUSSO" variant applied to real biopsy data (R only)
clussocode.zip          Clean R reference implementation, mirrors src/core/ 1:1
Supplemental_Code.zip   Adjacent R codebase for Random CLUSSO real-data analysis
clussocode.Rproj        RStudio project file for the R sources
```

### `src/core/` — CLUSSO

| File | Purpose |
|---|---|
| `Mainfunction_albet.py` | Alternating optimizer: fixes `alpha`, solves sparse `beta` via lasso; fixes `beta`, solves `alpha` via OLS; repeats to convergence. `_glmnet_lasso` reimplements R `glmnet`'s centering/scaling so results match the R version exactly. |
| `mat_vec_prd.py` | Tensor x vector products used in each alternation step. |
| `K_prdu.py` | Kronecker product helper, used for the convergence check `‖β⊗α − β0⊗α0‖`. |
| `coefficient.py` | Standalone glmnet-equivalent lasso path fitter with BIC lambda selection (used by the naive/average baseline). |
| `CLUSSO_Functions_Project1_6_16_23.py` | Simulation machinery: synthetic data generation + GMM clustering, TPR/FPR/L1-bias/MSE scoring, and `CLUSSO_performance` which benchmarks CLUSSO vs. naive averaging vs. the "full information" oracle. |
| `CLUSSO_Simulations_Project1_6_16_23.py` | Batch driver over a large parameter grid (noise, sparsity, `n`, `q`); one job per grid row, 1000 reps each, writes CSV results. |
| `CLUSSO_Data_Example.py` | End-to-end worked example: synthetic data -> GMM clustering -> CLUSSO fit vs. naive baseline -> printed coefficients and MSE. |

### `src/tepig/` — TEPIG

| File | Purpose |
|---|---|
| `simulation_synthetic.py` | Full synthetic simulation study on `(cluster, feature, slide)` tensors. Compares `tepig` (proximal-gradient/FISTA + group lasso, several thresholding variants), `tepig_lowrank` (rank-1 alternating, reference only), `clusso` (slides pooled), `naive`, and `oracle`. Run with `--n --q --sparsity`. |
| `simulation_bootstrap.py` | Same estimator family, but bootstraps from real pre-clustered subject data instead of generating fully synthetic data, for a more realistic noise/covariance structure. |
| `summarize_synthetic.py` | Aggregates the per-setting `.pkl` outputs of `simulation_synthetic.py` across the full `(n, q, sparsity)` grid into one comparison table. |

`simulation_bootstrap.py` and `simulation_synthetic.py` locate `src/core/` and
the output directory relative to their own path (`sys.path` includes
`../core`; outputs are written to `../../outputs`), so they must be run from
inside `src/tepig/` with `src/core/` as a sibling — the layout above is
required, not just a suggestion.

### R implementation

- `clussocode.zip` — clean R reference implementation matching `src/core/`
  file-for-file (`Mainfunction_albet.r`, `coefficient.r`, `K_prdu.r`,
  `mat_vec_prd.r`, `SLasso_MSE.R`, `CLUSSO_Functions_Project1_6_16_23.R`,
  `CLUSSO_Data_Example.R`).
- `Supplemental_Code.zip` / `Random_CLUSSO_*.R` — a related "Random CLUSSO"
  variant applied to real biopsy data (reads `ptvars.sas7bdat` /
  `tubulefeatures.sas7bdat`, outcome `eGFRatBx`). This variant and its
  Flex-Threshold/Structured-OLS `Mainfunction_albet` variants exist only in R;
  there is no TEPIG (tensor) equivalent.

## Known gap

`CLUSSO_Data_Example.py` and `CLUSSO_Functions_Project1_6_16_23.py` both
`import SLasso_MSE`, but `SLasso_MSE.py` does not exist anywhere in this
repo — only `SLasso_MSE.R` is present (inside `clussocode.zip`). Those two
files will not run until `SLasso_MSE.py` is ported from the R version and
added to `src/core/`.

## Running

```bash
# Worked example: synthetic data, CLUSSO vs. naive baseline
cd src/core && python CLUSSO_Data_Example.py

# Batch CLUSSO simulation grid (1-indexed job selects one grid row)
cd src/core && python CLUSSO_Simulations_Project1_6_16_23.py <job_index>

# TEPIG synthetic simulation study
cd src/tepig && python simulation_synthetic.py --n 300 --q 10 --sparsity 0.8

# Aggregate TEPIG synthetic results across the full grid
cd src/tepig && python summarize_synthetic.py

# TEPIG bootstrap simulation from real pre-clustered data
cd src/tepig && python simulation_bootstrap.py
```

Requires: `numpy`, `scikit-learn`, `pandas`, `joblib`.
