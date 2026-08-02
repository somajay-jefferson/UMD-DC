# Author: Jeremy Rubin
# Date: 6/16/23 (original), 2024 (Python port)
# Code to run CLUSSO simulations
#
# Usage:
#   python CLUSSO_Simulations_Project1_6_16_23.py <job_index>
#
# <job_index> selects one row from the parameter grid (1-indexed, matching R).
# Output CSVs are written to:
#   CLUSSO/median_stats/
#   CLUSSO/lambdas/
#   CLUSSO/extra_stats/

import sys
import os
import time
import numpy as np
import pandas as pd

from CLUSSO_Functions_Project1_6_16_23 import CLUSSO_performance

# ─────────────────────────────────────────────────────────────────────────────
# Fixed parameters
# ─────────────────────────────────────────────────────────────────────────────

mu          = 40
sigma_sq_m  = 5
sigma1      = 1
sigma2      = 3
x1          = 2
x2          = 5
sigma_sq    = 1
P           = 2
lambda_grid = np.arange(0.5, 5.1, 0.5)
nsim        = 1000

np.random.seed(83022)

# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid  (mirrors R's expand.grid calls)
# ─────────────────────────────────────────────────────────────────────────────

rows_main = [
    (1, s, nv, qv)
    for s  in [0.4, 0.8]
    for nv in [300, 500, 700, 900, 1100, 1500, 2000]
    for qv in [10, 50, 100, 150, 200]
]

rows_extra = [
    (sr, 0.8, 500, 10)
    for sr in range(1, 51, 5)
]

param_grid = rows_main + rows_extra

# ─────────────────────────────────────────────────────────────────────────────
# Read job index from command line (1-indexed, matching R)
# ─────────────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python CLUSSO_Simulations_Project1_6_16_23.py <job_index>")
    print(f"       job_index in 1..{len(param_grid)}")
    sys.exit(1)

j = int(sys.argv[1]) - 1    # convert to 0-indexed
if j < 0 or j >= len(param_grid):
    print(f"job_index must be between 1 and {len(param_grid)}")
    sys.exit(1)

sigma_r, s2, n, q = param_grid[j]
sigma_r = float(sigma_r)
n       = int(n)
q       = int(q)

print(f"Job {j+1}/{len(param_grid)}: sigma_r={sigma_r}, s={s2}, n={n}, q={q}")

# ─────────────────────────────────────────────────────────────────────────────
# True coefficients
# ─────────────────────────────────────────────────────────────────────────────

alpha_star = np.random.randint(1, 5, size=P).astype(float)
beta_star  = np.random.randint(1, 5, size=q).astype(float)

if s2 > 0:
    zero_idx            = np.random.choice(q, size=int(np.floor(s2 * q)), replace=False)
    beta_star[zero_idx] = 0.0

# True sparsity (may differ slightly from s2 due to rounding)
s2 = round(1.0 - (beta_star != 0).sum() / q, 3)

# Per-subject atrophic tubule weights
w = np.random.choice(np.arange(0.2, 0.9, 0.1), size=n, replace=True)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

time_vec  = np.zeros(nsim)
clust_vec = np.zeros(nsim)

CLUSSO_stat_mat = np.zeros((nsim, 4))
Avg_stat_mat    = np.zeros((nsim, 4))
Full_stat_mat   = np.zeros((nsim, 4))
lambda_mat      = np.zeros((nsim, 2))

for i in range(nsim):
    np.random.seed(i + 1)   # match R: set.seed(i), 1-indexed
    print(f"  Simulation {i+1}/{nsim}")

    t0     = time.time()
    result = CLUSSO_performance(n, mu, sigma_sq_m, sigma_sq,
                                 x1, x2, sigma1, sigma2, sigma_r,
                                 alpha_star, beta_star, lambda_grid, w)
    elapsed = time.time() - t0

    # Retry until we get a valid (non-None) result
    while result[0] is None:
        t0      = time.time()
        result  = CLUSSO_performance(n, mu, sigma_sq_m, sigma_sq,
                                      x1, x2, sigma1, sigma2, sigma_r,
                                      alpha_star, beta_star, lambda_grid, w)
        elapsed = time.time() - t0

    perf, lambdas, clust_acc = result

    time_vec[i]           = elapsed
    clust_vec[i]          = clust_acc
    lambda_mat[i, :]      = lambdas
    CLUSSO_stat_mat[i, :] = perf['CLUSSO'].values
    Avg_stat_mat[i, :]    = perf['Average balancing'].values
    Full_stat_mat[i, :]   = perf['Full information'].values

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate results
# ─────────────────────────────────────────────────────────────────────────────

median_mat = pd.DataFrame(
    np.column_stack([
        np.median(CLUSSO_stat_mat, axis=0),
        np.median(Avg_stat_mat,    axis=0),
        np.median(Full_stat_mat,   axis=0),
    ]),
    index   = ['TPR', 'FPR', 'L1 norm of bias', 'MSE'],
    columns = ['CLUSSO', 'Average balancing', 'Full information']
)

lambda_df = pd.DataFrame(
    lambda_mat,
    columns = ['CLUSSO', 'Full information']
)

extra_stats = pd.Series(
    [time_vec.mean(), clust_vec.mean() * 100],
    index = ['Average time taken per simulation', 'Average classification accuracy']
)

print(f"\nAverage time per simulation: {time_vec.mean():.2f}s")
print(f"Average clustering accuracy: {clust_vec.mean() * 100:.2f}%")
print("\nMedian statistics:")
print(median_mat)

# ─────────────────────────────────────────────────────────────────────────────
# Write output CSVs
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("CLUSSO/median_stats", exist_ok=True)
os.makedirs("CLUSSO/lambdas",      exist_ok=True)
os.makedirs("CLUSSO/extra_stats",  exist_ok=True)

file_tag = f"sigma_r_{sigma_r}_s_{s2}_n_{n}_q_{q}"

median_mat.to_csv( f"CLUSSO/median_stats/6_16_23_full_CLUSSO_performance{file_tag}.csv")
lambda_df.to_csv(  f"CLUSSO/lambdas/6_16_23_full_CLUSSO_lambdas_{file_tag}.csv")
extra_stats.to_csv(f"CLUSSO/extra_stats/6_16_23_full_CLUSSO_extra_stats_{file_tag}.csv")

print(f"\nResults written with tag: {file_tag}")
