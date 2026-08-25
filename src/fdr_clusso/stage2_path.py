# Stage 2 gate: how does CLUSSO's support size respond to lambda?
#
# PS-Fdr's step 2 pins the number of variables the permuted arm may select, so
# the null cannot quietly collapse to nothing and drag the estimated false
# discovery count down with it. With a lasso that is free: `lars_path` yields
# the whole coefficient path in one solve and you read off the point where k
# variables are active (ps_fdr.lasso_support_fixed_k).
#
# CLUSSO has no path. Every lambda is a separate alternating solve, and because
# the objective is non-convex the map from lambda to support size is not
# guaranteed monotone -- different lambdas can drop the optimizer into different
# valleys with unrelated supports. Nothing in this repo has ever measured that
# map, so the claim on docs/fdr-clusso.html that it misbehaves is, as things
# stand, an assertion.
#
# What is actually needed is weaker than the docs page assumes. The failure the
# fixed count prevents is the permuted arm selecting too FEW; overshoot only
# makes the null stronger, which is the safe direction. So `>= k` suffices, and
# `>= k` tolerates non-monotonicity -- a descending scan just needs the support
# to reach k somewhere on the grid.
#
# Gate:
#   monotone (or inversions bounded and small)  -> descending scan ships as-is
#   target k unreachable on > 10% of pairs      -> widen the grid, then reconsider
#
# Usage:
#   python stage2_path.py --datasets 20
#   python stage2_path.py --quick

import argparse
import json

import numpy as np

from clusso_select import (DEFAULT_LAMBDA_GRID, DEFAULT_N_STARTS,
                           clusso_fit_cv, clusso_fit_multistart,
                           support_mask, threshold_binding)
from stage1_noise import make_cohort


def path_for(X, y, grid, n_starts=DEFAULT_N_STARTS):
    """Support size at every lambda, scanned sparse to dense."""
    lams = np.sort(np.asarray(grid, dtype=float))[::-1]
    ks, binding = [], []
    for lam in lams:
        fit = clusso_fit_multistart(X, y, lam, n_starts=n_starts)
        ks.append(int(support_mask(fit['bet']).sum()))
        binding.append(threshold_binding(fit['bet']))
    return lams, np.array(ks), np.array(binding)


def inversions(ks):
    """
    Where the support GROWS as the penalty rises -- i.e. scanning sparse to
    dense, where k goes down instead of up. Returns (count, worst drop).
    """
    d = np.diff(ks)
    bad = d[d < 0]
    return int(bad.size), int(-bad.min()) if bad.size else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=int, default=20)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--q', type=int, default=50)
    ap.add_argument('--sparsity', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=20260817)
    ap.add_argument('--grid', type=int, default=len(DEFAULT_LAMBDA_GRID))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()

    if args.quick:
        args.datasets, args.n, args.q, args.grid = 5, 150, 30, 18

    grid = np.geomspace(0.01, 20.0, args.grid)
    print(f'{args.datasets} cohorts, n={args.n} q={args.q}, '
          f'{args.grid}-point grid [{grid[0]:g}, {grid[-1]:g}], '
          f'n_starts={DEFAULT_N_STARTS}')
    print()

    rows = []
    for d in range(args.datasets):
        X, y, truth, acc = make_cohort(args.n, args.q, args.sparsity,
                                       args.seed + d)
        lams, ks, binding = path_for(X, y, grid)
        n_inv, worst = inversions(ks)
        lam_cv = clusso_fit_cv(X, y, lambda_grid=grid)['lam']
        k_cv = int(ks[np.argmin(np.abs(lams - lam_cv))])

        rows.append({'dataset': d, 'k_min': int(ks.min()), 'k_max': int(ks.max()),
                     'n_inversions': n_inv, 'worst_inversion': worst,
                     'monotone': n_inv == 0, 'lam_cv': float(lam_cv),
                     'k_cv': k_cv, 'binding_frac': float(binding.mean()),
                     'ks': ks.tolist()})

        flag = 'monotone' if n_inv == 0 else f'{n_inv} inversions, worst -{worst}'
        print(f'  d{d:<2d} k spans {ks.min():3d}..{ks.max():3d}   '
              f'lam_cv {lam_cv:7.4f} -> k {k_cv:3d}   '
              f'0.001-cut live on {binding.mean():4.0%} of the path   {flag}')

    n_mono = sum(r['monotone'] for r in rows)
    worst_all = max(r['worst_inversion'] for r in rows)
    k_max_all = [r['k_max'] for r in rows]
    k_min_all = [r['k_min'] for r in rows]

    print()
    print(f'  monotone: {n_mono}/{len(rows)}   worst inversion anywhere: '
          f'-{worst_all}')
    print(f'  grid reaches k={min(k_min_all)}..{max(k_max_all)} '
          f'(q={args.q}); sparsest end min k = {max(k_min_all)}')

    # Reachability: PS-Fdr pins the permuted arm at the real arm's median
    # support size, so the question is whether a descending scan can reach the
    # k values that will actually be asked for.
    targets = sorted({r['k_cv'] for r in rows})
    unreachable = [(r['dataset'], t) for r in rows for t in targets
                   if t > r['k_max']]
    frac_unreachable = len(unreachable) / max(len(rows) * len(targets), 1)
    print(f'  target k values seen: {targets}')
    print(f'  unreachable (dataset, k) pairs: {len(unreachable)}/'
          f'{len(rows) * len(targets)} = {frac_unreachable:.1%}')

    print('\n' + '=' * 68)
    if n_mono == len(rows):
        print('  PASS  the map is monotone on every cohort. A descending scan')
        print('        hits any reachable target on the first crossing.')
    elif worst_all <= 2:
        print(f'  PASS  {len(rows) - n_mono} cohorts show inversions, but the worst')
        print(f'        is {worst_all} feature(s). ">= k" tolerates overshoot, so a')
        print('        descending scan is still sound. Exact-k would not be.')
    else:
        print(f'  MARGINAL  worst inversion is {worst_all} features. ">= k" still')
        print('        works -- it only needs to reach the bar, not land on it --')
        print('        but exact-k is definitively out.')
    if frac_unreachable > 0.10:
        print(f'  FAIL  {frac_unreachable:.1%} of targets are off the grid. Widen the')
        print('        sparse end, or the null arm will silently under-select and')
        print('        Fdr-hat will be anti-conservative.')
    else:
        print(f'  PASS  {frac_unreachable:.1%} of targets unreachable (gate: < 10%).')
    print('=' * 68)

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'args': vars(args), 'grid': grid.tolist(),
                       'rows': rows, 'n_monotone': n_mono,
                       'worst_inversion': worst_all,
                       'frac_unreachable': frac_unreachable}, fh, indent=2)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()
