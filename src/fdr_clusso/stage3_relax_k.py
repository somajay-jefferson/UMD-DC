# Stage 3 gate: what does it cost to let the permuted arm OVERSHOOT its target?
#
# PS-Fdr's step 2 pins the number of variables the permuted arm may select. A
# lasso hits an exact target for free -- lars_path gives the whole coefficient
# path in one solve. CLUSSO has no path, so it can clear a target size but
# cannot land on one, and stage 2 measured it overshooting: the map from lambda
# to support size inverts on 6 of 20 cohorts by up to 3 features, and on 4 of 20
# the sparse end of the grid still keeps 13 features when a target of 10 is
# asked for.
#
# WHAT THIS SCRIPT ORIGINALLY DID, AND WHY THAT WAS WRONG
#
# The plan was to run PS-Fdr with rule='exact' and rule='atleast' on the same
# cohorts and compare. That comparison is vacuous, and part A below proves it:
# LARS adds one variable at a time, so nnz increments by 0 or 1 and the path
# always lands EXACTLY on k, never past it. Measured over 120 (cohort, k) pairs:
# the path stepped past k zero times, and the three rules returned an identical
# support every single time.
#
# So on a lasso, "at least k" IS "exactly k". The rule change is a no-op, and
# comparing the rules cannot say anything about it. The first version of this
# script reported PASS with all three rules identical to three decimal places,
# which is a test passing because it measures nothing.
#
# WHAT IT DOES INSTEAD
#
# The rules only diverge when the selector overshoots, which a lasso never does
# and CLUSSO does. So price the overshoot directly: hold everything else fixed
# and inflate the permuted arm's target size by delta, using ps_fdr's k_null
# override. delta = 0 is the published rule; delta > 0 is what CLUSSO's floor
# would produce.
#
# This is still lasso-only and still cheap, and it asks the question that
# actually matters: when the null arm is systematically wider than the paper
# intends, does FDR control survive, and what does it cost in power?
#
# Expected direction: a wider null arm has higher selection frequencies, so e0
# rises, so Fdr-hat rises, so fewer variables clear the bar -- LOWER FDP and
# LOWER power. Giving up power to protect the guarantee is the safe way to be
# wrong. If FDP goes UP instead, the floor is unsafe and CLUSSO cannot use it.
#
# Gate at q = 0.1, against delta = 0, at the overshoot stage 2 measured (+3):
#   mean FDP   must not rise by more than 0.02
#   mean power must not fall by more than 0.05
#
# Usage:
#   python stage3_relax_k.py --reps 200
#   python stage3_relax_k.py --quick

import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'fdr'))

from ps_fdr import (_standardize, fdp_power, lasso_support_fixed_k,  # noqa: E402
                    ps_fdr)
from PS_Fdr_Data_Example import BETA_STAR, make_data                 # noqa: E402

Q = 0.1
DELTAS = (0, 1, 2, 3, 5, 8)


def part_a_rules_coincide(n_cohorts=40, ks=(12, 18, 24), seed=5):
    """
    Evidence that comparing exact/atleast/nearest on a lasso measures nothing.
    Reported rather than assumed, because it is the reason this script does not
    do what the plan said it would.
    """
    from sklearn.linear_model import lars_path
    rng = np.random.default_rng(seed)
    total = steps_past = differ = 0

    for _ in range(n_cohorts):
        X, y = make_data(rng)
        idx = rng.integers(0, X.shape[0], size=X.shape[0])
        Xb, yb = X[idx], y[idx]
        Xs, ys, _ = _standardize(Xb, yb)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, _, coefs = lars_path(Xs, ys, method='lasso', return_path=True)
        nnz = (coefs != 0).sum(axis=0)

        for k in ks:
            total += 1
            reached = np.flatnonzero(nnz >= k)
            if reached.size and nnz[reached[0]] > k:
                steps_past += 1
            masks = [lasso_support_fixed_k(Xb, yb, k, rule=r)
                     for r in ('exact', 'atleast', 'nearest')]
            if not all(np.array_equal(masks[0], m) for m in masks[1:]):
                differ += 1

    return {'pairs': total, 'steps_past_k': steps_past, 'rules_differ': differ}


def cohort_all_deltas(seed, deltas, B, M, q=Q):
    """
    One cohort, every delta. The real arm is identical across deltas -- only the
    permuted arm's target moves -- so the baseline fit is done once and its k
    reused, rather than recomputed per delta.
    """
    rng = np.random.default_rng(seed)
    X, y = make_data(rng)
    truth = BETA_STAR != 0

    base = ps_fdr(X, y, q=q, B=B, M=M, seed=seed, null_mode='fixed')
    k = int(base['k'])

    out = {}
    for d in deltas:
        res = base if d == 0 else ps_fdr(X, y, q=q, B=B, M=M, seed=seed,
                                         null_mode='fixed', k_null=k + d)
        fdp, power = fdp_power(res['selected'], truth)
        out[d] = {'fdp': fdp, 'power': power,
                  'n_sel': int(res['selected'].size),
                  'k': k, 'k_used': int(res['k_used'])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--B', type=int, default=50)
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=20260826)
    ap.add_argument('--gate-delta', type=int, default=3,
                    help='the overshoot the gate is applied at; stage 2 '
                         'measured CLUSSO at up to 3 features')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()

    if args.quick:
        args.reps, args.B, args.M = 25, 25, 40

    from joblib import Parallel, delayed

    print('part A -- do exact / atleast / nearest ever differ on a lasso path?')
    a = part_a_rules_coincide(n_cohorts=10 if args.quick else 40)
    print(f'  {a["pairs"]} (cohort, k) pairs   path stepped past k: '
          f'{a["steps_past_k"]}   rules gave a different support: '
          f'{a["rules_differ"]}')
    if a['rules_differ'] == 0:
        print('  -> identical every time. On a lasso "at least k" IS "exactly k",')
        print('     so comparing the rules measures nothing. Hence part B.\n')
    else:
        print('  -> the rules DO differ here; the direct comparison is live '
              'after all.\n')

    truth = BETA_STAR != 0
    seeds = [args.seed + i for i in range(args.reps)]
    print(f'part B -- price the overshoot directly. {args.reps} cohorts x '
          f'{len(DELTAS)} deltas, B={args.B} M={args.M}, q={Q}')
    print(f'  {int(truth.sum())} real features of {truth.size}; '
          f'delta inflates the permuted arm target above the real arm median\n')

    per_cohort = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(cohort_all_deltas)(s, DELTAS, args.B, args.M) for s in seeds)

    print(f'  {"delta":>6} {"k used":>7} {"mean FDP":>9} {"power":>7} '
          f'{"mean |S|":>9} {"FDP<=q":>7}')
    out = {}
    for d in DELTAS:
        runs = [c[d] for c in per_cohort]
        fdp = np.array([r['fdp'] for r in runs])
        pw = np.array([r['power'] for r in runs])
        ns = np.array([r['n_sel'] for r in runs])
        ku = np.array([r['k_used'] for r in runs])
        out[d] = {'fdp': float(fdp.mean()), 'power': float(pw.mean()),
                  'n_sel': float(ns.mean()), 'k_used': float(np.median(ku)),
                  'hit_rate': float(np.mean(fdp <= Q)),
                  'fdp_sd': float(fdp.std(ddof=1)),
                  'power_sd': float(pw.std(ddof=1))}
        r = out[d]
        print(f'  {d:>6} {r["k_used"]:>7.0f} {r["fdp"]:>9.3f} '
              f'{r["power"]:>7.3f} {r["n_sel"]:>9.2f} {r["hit_rate"]:>7.3f}')

    g = args.gate_delta
    d_fdp = out[g]['fdp'] - out[0]['fdp']
    d_pw = out[g]['power'] - out[0]['power']

    print('\n' + '=' * 70)
    print(f'  gate at delta = {g} (the overshoot stage 2 measured for CLUSSO)')
    print(f'    FDP {out[0]["fdp"]:.3f} -> {out[g]["fdp"]:.3f}  ({d_fdp:+.3f})')
    print(f'    power {out[0]["power"]:.3f} -> {out[g]["power"]:.3f}  ({d_pw:+.3f})')

    ok_fdp, ok_pw = d_fdp <= 0.02, d_pw >= -0.05
    if ok_fdp and ok_pw:
        print('  PASS  overshooting the target does not break FDR control and')
        print('        does not cost more than 5 points of power. A floor is')
        print('        safe for CLUSSO to use; stage 4 may proceed.')
    elif not ok_fdp:
        print(f'  FAIL  overshoot RAISES false discoveries by {d_fdp:+.3f} '
              f'(gate <= +0.02).')
        print('        Wrong direction, and not merely a power cost. CLUSSO')
        print('        cannot use a floor.')
    else:
        print(f'  FAIL  overshoot costs {-d_pw:.3f} of power (gate <= 0.05).')
        print('        Valid but too expensive. Tighten the lambda grid so the')
        print('        floor lands closer, or reconsider the regime.')

    if d_fdp > 0:
        print(f'\n  NOTE  FDP moved UP ({d_fdp:+.3f}). A wider null arm was')
        print('        expected to be conservative. Worth explaining before')
        print('        relying on it, even if the gate passes.')
    print('=' * 70)

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'args': vars(args), 'q': Q, 'part_a': a,
                       'deltas': {str(k): v for k, v in out.items()},
                       'gate_delta': g, 'delta_fdp': d_fdp,
                       'delta_power': d_pw,
                       'pass': bool(ok_fdp and ok_pw)}, fh, indent=2)
        print(f'wrote {args.json}')

    return 0 if (ok_fdp and ok_pw) else 1


if __name__ == '__main__':
    sys.exit(main())
