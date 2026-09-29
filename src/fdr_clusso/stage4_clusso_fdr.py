# Stage 4: the composition itself. PS-Fdr with CLUSSO as the selection rule.
#
# Stages 1-3b established the pieces on a lasso, where both arms' rules can be
# run and compared. This is the first script that puts CLUSSO inside PS-Fdr and
# asks the question the whole directory exists for: does the composition control
# the false discovery rate, and what does it cost in power?
#
# The comparison that makes the number mean something is the BASELINE arm --
# CV-tuned CLUSSO with no FDR control, which is CLUSSO exactly as it ships. That
# is the status quo the wrapper has to beat. If FDR-CLUSSO's FDP is not lower
# than plain CLUSSO's, the wrapper bought nothing and the answer is no.
#
# TWO NULL-ARM RULES, run on identical cohorts:
#
#   arm A   null_mode='fixed', null_rule='exact'
#           The published rule, ported. Scan lambda sparse -> dense, stop at the
#           first support of at least k, then keep the k largest coefficients.
#           The truncation is not an approximation -- it is what
#           ps_fdr.lasso_support_fixed_k already does when a path step overshoots
#           (ps_fdr.py:112-113). Porting it is what makes exact k reachable for
#           an estimator with no coefficient path, and it retires stage 3's
#           overshoot cost (power 0.688 -> 0.565) by removing the overshoot.
#
#   arm B   null_mode='fixed_lambda'
#           One CV-tuned lambda, chosen on the real data before any resampling,
#           used for every fit on both arms. Nothing targets a count, so nothing
#           can miss one. Passed on the plain lasso (stage3b.json) but it is a
#           DEVIATION from the published procedure, so it is the comparison arm
#           and not the headline. If the arms agree, that is reassuring. If they
#           diverge, the gap measures how much the count-fixing device was doing.
#
# FROZEN CLUSTERING. `make_cohort` builds the (P, q, n) design tensor once, per
# cohort, before anything resamples. The clustering does not happen again inside
# the loop. Every number this script produces is therefore CONDITIONAL on one
# clustering solution -- see REVIEW.md section 4. Stage 5 owes the second arm
# that prices what that costs, using the true labels the simulation knows.
#
# THREE DIAGNOSTICS, reported rather than merely computed:
#
#   exhausted_rate   The one failure truncation does NOT fix. If the lambda grid
#                    runs out before reaching k, the permuted support comes back
#                    short, e0 is too small and Fdr-hat is too small. That is
#                    ANTI-CONSERVATIVE and silent. A material rate here means
#                    arm A's number is not trustworthy, and the gate says so.
#   null_top         Arm B's failure mode: a null arm narrower than the real one
#                    over-selects. stage3b.json's lasso reference is 0.928.
#   threshold_frac   How often core's hardcoded 0.001 cut is plausibly deciding
#                    the support rather than the lasso. REVIEW.md section 6
#                    measured 52-85% at this exact n=300, q=50 setting. If it is
#                    deciding the support, it is deciding the result.

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'fdr'))

from ps_fdr import fdp_power                              # noqa: E402

from clusso_ps_fdr import ClussoSelector, clusso_ps_fdr   # noqa: E402
from clusso_select import (clusso_fit_cv, support_mask,   # noqa: E402
                           threshold_binding)
from stage1_noise import make_cohort                      # noqa: E402

Q = 0.1

ARMS = {
    'exact_k':      {'null_mode': 'fixed',        'null_rule': 'exact'},
    'fixed_lambda': {'null_mode': 'fixed_lambda', 'null_rule': 'exact'},
}


def one_cohort(seed, B, M, n, q_feat, sparsity, arms, q=Q):
    # The clustering happens HERE, once, and never again. Everything downstream
    # resamples subjects out of this one frozen tensor.
    clust_X, Y, truth, clust_acc = make_cohort(n, q_feat, sparsity, seed)

    out = {'clust_acc': float(clust_acc), 'n_true': int(truth.sum())}

    # Baseline: CLUSSO as it ships. CV-tuned, no FDR control, one fit.
    base_fit = clusso_fit_cv(clust_X, Y, n_folds=5, rng=None, n_starts=5)
    base_mask = support_mask(base_fit['bet'])
    b_fdp, b_pw = fdp_power(np.flatnonzero(base_mask), truth)
    out['baseline'] = {'fdp': float(b_fdp), 'power': float(b_pw),
                       'n_sel': int(base_mask.sum()),
                       'lam_cv': float(base_fit['lam']),
                       'thresh_binding': bool(threshold_binding(base_fit['bet']))}

    for arm in arms:
        sel = ClussoSelector()
        res, sel = clusso_ps_fdr(clust_X, Y, q=q, B=B, M=M, seed=seed,
                                 selector=sel, **ARMS[arm])
        fdp, pw = fdp_power(res['selected'], truth)
        out[arm] = {
            'fdp': float(fdp),
            'power': float(pw),
            'n_sel': int(len(res['selected'])),
            'k': int(res['k']),
            'lam_star': (float(res['lam_star'])
                         if res['lam_star'] is not None else None),
            'null_top': float(res['Pi_bar'][-1]),
            'fdr_hat': (float(res['fdr_hat'])
                        if np.isfinite(res['fdr_hat']) else None),
            'exhausted_rate': float(sel.exhausted_rate),
            'mean_fits': float(sel.mean_fits_per_target),
            'n_truncated': int(sel.n_truncated),
            'n_fixed_k_calls': int(sel.n_fixed_k_calls),
        }
    return out


def _agg(runs, key):
    return np.array([r[key] for r in runs], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--B', type=int, default=50)
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--q-features', type=int, default=50)
    ap.add_argument('--sparsity', type=float, default=0.8)
    ap.add_argument('--arm', choices=('a', 'b', 'both'), default='both')
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--exhausted-max', type=float, default=0.02,
                    help='gate: max tolerable rate of lambda-grid exhaustion '
                         'on the permuted arm, since that error is '
                         'anti-conservative and silent')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()

    if args.quick:
        args.reps, args.B, args.M, args.n, args.q_features = 3, 10, 10, 120, 25

    arms = {'a': ['exact_k'], 'b': ['fixed_lambda'],
            'both': ['exact_k', 'fixed_lambda']}[args.arm]

    from joblib import Parallel, delayed

    seeds = [args.seed + i for i in range(args.reps)]
    print(f'{args.reps} cohorts, n={args.n} q={args.q_features} '
          f'sparsity={args.sparsity}, B={args.B} M={args.M}, target q={Q}')
    print('  CLUSSO inside PS-Fdr, clustering frozen once per cohort\n')

    per_cohort = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(one_cohort)(s, args.B, args.M, args.n, args.q_features,
                            args.sparsity, arms)
        for s in seeds)

    out = {}
    print(f'  {"arm":>14} {"mean FDP":>9} {"power":>7} {"mean |S|":>9} '
          f'{"FDP<=q":>7} {"null top":>9} {"exhaust":>8}')

    base = [c['baseline'] for c in per_cohort]
    b_fdp, b_pw = _agg(base, 'fdp'), _agg(base, 'power')
    out['baseline'] = {'fdp': float(b_fdp.mean()), 'power': float(b_pw.mean()),
                       'n_sel': float(_agg(base, 'n_sel').mean()),
                       'hit_rate': float(np.mean(b_fdp <= Q)),
                       'fdp_sd': (float(b_fdp.std(ddof=1))
                                  if len(b_fdp) > 1 else 0.0),
                       'thresh_binding': float(
                           np.mean([r['thresh_binding'] for r in base]))}
    r = out['baseline']
    print(f'  {"CLUSSO (bare)":>14} {r["fdp"]:>9.3f} {r["power"]:>7.3f} '
          f'{r["n_sel"]:>9.2f} {r["hit_rate"]:>7.3f} {"-":>9} {"-":>8}')

    for arm in arms:
        runs = [c[arm] for c in per_cohort]
        fdp, pw = _agg(runs, 'fdp'), _agg(runs, 'power')
        out[arm] = {
            'fdp': float(fdp.mean()), 'power': float(pw.mean()),
            'n_sel': float(_agg(runs, 'n_sel').mean()),
            'k': float(_agg(runs, 'k').mean()),
            'hit_rate': float(np.mean(fdp <= Q)),
            'null_top': float(_agg(runs, 'null_top').mean()),
            'exhausted_rate': float(_agg(runs, 'exhausted_rate').mean()),
            'mean_fits': float(_agg(runs, 'mean_fits').mean()),
            'fdp_sd': float(fdp.std(ddof=1)) if len(fdp) > 1 else 0.0,
        }
        r = out[arm]
        print(f'  {arm:>14} {r["fdp"]:>9.3f} {r["power"]:>7.3f} '
              f'{r["n_sel"]:>9.2f} {r["hit_rate"]:>7.3f} '
              f'{r["null_top"]:>9.3f} {r["exhausted_rate"]:>8.3f}')

    out['clust_acc'] = float(np.mean([c['clust_acc'] for c in per_cohort]))

    # --- the gate ---------------------------------------------------------
    # Arm A is the published rule, so arm A is what gates. Arm B is reported
    # beside it and does not.
    head = 'exact_k' if 'exact_k' in arms else arms[0]
    a, b = out[head], out['baseline']

    print('\n' + '=' * 78)
    print(f'  Gate reads arm {head!r} -- the ported rule. Target q = {Q}.\n')

    ok_fdr = a['fdp'] <= Q
    beats_base = a['fdp'] < b['fdp']
    has_power = a['power'] > 0.0
    ok_exh = a['exhausted_rate'] <= args.exhausted_max

    print(f'    FDP <= q              {a["fdp"]:.3f} <= {Q}        '
          f'{"PASS" if ok_fdr else "FAIL"}')
    print(f'    beats bare CLUSSO     {a["fdp"]:.3f} <  {b["fdp"]:.3f}      '
          f'{"PASS" if beats_base else "FAIL"}')
    print(f'    selects something     power {a["power"]:.3f} vs {b["power"]:.3f} '
          f'bare  {"PASS" if has_power else "FAIL"}')
    print(f'    grid not exhausted    {a["exhausted_rate"]:.4f} <= '
          f'{args.exhausted_max}     {"PASS" if ok_exh else "FAIL"}')

    passed = ok_fdr and beats_base and has_power and ok_exh
    print(f'\n  => {"PASS" if passed else "FAIL"}')

    if not ok_exh:
        print('\n  The exhaustion failure is the serious one. A permuted arm that')
        print('  came back SHORT of k makes e0 too small and Fdr-hat too small,')
        print('  so the FDP above is optimistic by an unknown amount. Widen')
        print('  DEFAULT_LAMBDA_GRID at the dense end before believing anything.')
    if ok_fdr and not beats_base:
        print('\n  FDR is controlled but bare CLUSSO already controlled it here.')
        print('  The wrapper is not being asked a hard enough question -- push')
        print('  --q-features up, where CLUSSO false-positives at 20-30%.')

    if 'fixed_lambda' in out and 'exact_k' in out:
        fl, ek = out['fixed_lambda'], out['exact_k']
        print(f'\n  arm A vs arm B:  FDP {ek["fdp"]:.3f} vs {fl["fdp"]:.3f}   '
              f'power {ek["power"]:.3f} vs {fl["power"]:.3f}')
        print('    Agreement means the count-fixing device was not load-bearing')
        print('    for CLUSSO. Divergence measures how much it was doing.')

    print(f'\n  cost: {a["mean_fits"]:.1f} CLUSSO fits per permuted-arm target')
    print(f'  diagnostic: core 0.001 threshold plausibly binding on '
          f'{b["thresh_binding"]:.0%} of cohorts')
    print(f'  clustering accuracy {out["clust_acc"]:.3f} '
          f'(every number above is conditional on it)')
    print('=' * 78)

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'args': vars(args), 'q': Q, 'arms': out,
                       'per_cohort': per_cohort, 'pass': bool(passed)},
                      fh, indent=1)
        print(f'wrote {args.json}', file=sys.stderr)

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
