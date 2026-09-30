# Does one lambda, used everywhere, work as well as pinning the count?
#
# PS-Fdr makes lambda do two incompatible jobs. On the real arm it is tuned by
# CV and means "predict well". On the permuted arm it is whatever hits a target
# support size -- a cardinality knob with nothing to do with prediction.
#
# A lasso gets away with that because the coefficient path decouples the two:
# one solve gives every lambda, so you read off either the best-CV lambda or the
# lambda giving k features from the same object, for free. CLUSSO has no path.
# Each use needs its own expensive search, and the second use additionally needs
# lambda to behave like a monotone dial -- which stage 2 measured it failing to
# do on 6 of 20 cohorts, by up to 3 features.
#
# So: stop making lambda do the second job. Tune ONE lambda on the real data,
# up front, and use it for every fit on both arms. Then lambda only ever means
# "predict well", nothing targets a support size, and nothing can overshoot one.
#
# What that would buy, if it holds:
#   - clusso_support_at_least_k becomes unnecessary
#   - stage 2's non-monotonicity stops mattering; nothing steers by size
#   - stage 3's overshoot cost disappears, because there is no overshoot
#   - far less compute: no CV inside the bootstrap, no lambda scan on the null
#
# What could break it, and it is exactly what the fixed count exists to prevent:
# the permuted arm's support size is now uncontrolled. With no signal to fit,
# the same penalty zeroes more coefficients, so the permuted arm may come out
# systematically narrower than the real arm. Narrower null -> smaller e0 ->
# smaller Fdr-hat -> over-selection. That is the anti-conservative failure.
#
# It is NOT the same as the variant the paper kills. Under null_mode='cv' the
# permuted arm re-tunes and picks a huge penalty precisely because nothing
# predicts, selecting almost nothing. Fixing lambda at the real data's value
# does not let it do that. So this sits between the invalid variant and the
# paper's fix, and where exactly is an empirical question.
#
# Three arms on identical cohorts, plain lasso only:
#
#   fixed         permuted count pinned to the real arm's median   (the paper)
#   cv            permuted arm re-tunes by CV                       (known bad)
#   fixed_lambda  one lambda, tuned once, used on both arms         (the idea)
#
# Read FDP first. 'cv' is the calibration: it shows how bad an anti-conservative
# null looks on this data. If fixed_lambda tracks 'fixed', the idea works. If it
# drifts toward 'cv', it does not.
#
# Usage:
#   python stage3b_fixed_lambda.py --reps 120
#   python stage3b_fixed_lambda.py --quick

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'fdr'))

from ps_fdr import fdp_power, ps_fdr                       # noqa: E402
from PS_Fdr_Data_Example import BETA_STAR, make_data       # noqa: E402

Q = 0.1
# 'cv' is the paper's known-bad variant, kept as a calibration arm. It is by
# far the most expensive of the three -- a full LassoCV inside every one of the
# M*B permuted fits -- and its behaviour on this exact cohort is already
# reported on docs/ps-fdr.html (null top 0.801 vs 0.936, six features selected,
# realised FDP 0.167 against a promised 0.10). So it is off by default.
ALL_MODES = ('fixed', 'cv', 'fixed_lambda')
MODES = ('fixed', 'fixed_lambda')


def one_cohort(seed, B, M, modes, q=Q):
    rng = np.random.default_rng(seed)
    X, y = make_data(rng)
    truth = BETA_STAR != 0

    out = {}
    for mode in modes:
        res = ps_fdr(X, y, q=q, B=B, M=M, seed=seed, null_mode=mode)
        fdp, power = fdp_power(res['selected'], truth)
        out[mode] = {
            'fdp': fdp, 'power': power,
            'n_sel': int(res['selected'].size),
            'k': int(res['k']),
            # the permuted arm's own support sizes are what decide whether the
            # null is strong enough; record how wide it actually came out
            'null_top': float(res['Pi_bar'][-1]),
            'lam_star': res.get('lam_star'),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=120)
    ap.add_argument('--B', type=int, default=50)
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=20260902)
    ap.add_argument('--with-cv', action='store_true',
                    help="also run the paper's known-bad 'cv' arm as a "
                         'calibration reference (slow)')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()

    if args.quick:
        args.reps, args.B, args.M = 30, 25, 40

    from joblib import Parallel, delayed

    modes = ALL_MODES if args.with_cv else MODES
    truth = BETA_STAR != 0
    seeds = [args.seed + i for i in range(args.reps)]
    print(f'{args.reps} cohorts x {len(modes)} null modes, '
          f'B={args.B} M={args.M}, q={Q}')
    print(f'  {int(truth.sum())} real features of {truth.size}, plain lasso\n')

    per_cohort = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(one_cohort)(s, args.B, args.M, modes) for s in seeds)

    print(f'  {"null mode":>13} {"mean FDP":>9} {"power":>7} {"mean |S|":>9} '
          f'{"FDP<=q":>7} {"null top":>9}')
    out = {}
    for mode in modes:
        runs = [c[mode] for c in per_cohort]
        fdp = np.array([r['fdp'] for r in runs])
        pw = np.array([r['power'] for r in runs])
        ns = np.array([r['n_sel'] for r in runs])
        nt = np.array([r['null_top'] for r in runs])
        out[mode] = {'fdp': float(fdp.mean()), 'power': float(pw.mean()),
                     'n_sel': float(ns.mean()),
                     'hit_rate': float(np.mean(fdp <= Q)),
                     'null_top': float(nt.mean()),
                     'fdp_sd': float(fdp.std(ddof=1))}
        r = out[mode]
        print(f'  {mode:>13} {r["fdp"]:>9.3f} {r["power"]:>7.3f} '
              f'{r["n_sel"]:>9.2f} {r["hit_rate"]:>7.3f} {r["null_top"]:>9.3f}')

    f, fl = out['fixed'], out['fixed_lambda']
    c = out.get('cv')

    print('\n' + '=' * 72)
    print(f'  target q = {Q}.  FDP is the promise; power is what it costs.')
    print(f'    paper (fixed)        FDP {f["fdp"]:.3f}   power {f["power"]:.3f}')
    if c:
        print(f'    known-bad (cv)       FDP {c["fdp"]:.3f}   power {c["power"]:.3f}')
    print(f'    one lambda           FDP {fl["fdp"]:.3f}   power {fl["power"]:.3f}')
    print()

    # Where does fixed_lambda sit between the good and bad arms? 0 = on the
    # paper's rule, 1 = as bad as the variant the paper rejects.
    span = (c['fdp'] - f['fdp']) if c else 0.0
    pos = (fl['fdp'] - f['fdp']) / span if abs(span) > 1e-9 else float('nan')

    ok_fdp = fl['fdp'] <= Q and fl['fdp'] <= f['fdp'] + 0.02
    better_power = fl['power'] >= f['power'] - 0.05

    if ok_fdp and better_power:
        print('  PASS  one lambda holds FDR control and does not cost power.')
        print('        It removes the cardinality search entirely, which makes')
        print('        stage 2 moot and stage 3 unnecessary. Adopt it for CLUSSO.')
    elif not ok_fdp:
        print(f'  FAIL  one lambda breaks FDR control: FDP {fl["fdp"]:.3f} vs '
              f'{f["fdp"]:.3f} for the paper.')
        print(f'        It sits {pos:.0%} of the way from the paper rule to the')
        print('        variant the paper rejects. The permuted arm is too narrow;')
        print('        pinning the count is doing real work and must be kept.')
    else:
        print(f'  MIXED  FDR control holds but power drops '
              f'{f["power"] - fl["power"]:.3f}.')
        print('        Cheaper and simpler, but it detects less. Worth it only')
        print('        if the compute saving matters more than the detection.')

    print(f'\n  diagnostic: mean top-ranked null frequency')
    cv_bit = f'   cv {c["null_top"]:.3f}' if c else ''
    print(f'    fixed {f["null_top"]:.3f}{cv_bit}   one lambda {fl["null_top"]:.3f}')
    print('    a LOWER null top means a weaker bar, which is how the invalid')
    print('    variant over-selects. Compare one lambda against both.')
    print('=' * 72)

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'args': vars(args), 'q': Q, 'modes': out,
                       'position_between': pos,
                       'pass': bool(ok_fdp and better_power)}, fh, indent=2)
        print(f'wrote {args.json}')

    return 0 if (ok_fdp and better_power) else 1


if __name__ == '__main__':
    sys.exit(main())
