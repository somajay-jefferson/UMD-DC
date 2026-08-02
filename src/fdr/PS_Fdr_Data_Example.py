# Worked example for PS-Fdr, sized so every intermediate quantity fits on a
# page: 70 subjects, 30 features, 5 of which are real.
#
# Prints, in the order the algorithm runs them:
#   0  the data and the CV-tuned lasso baseline
#   1  bootstrap selection frequencies Pi_j
#   2  the permutation null, averaged over ranks
#   3  the SAM normalization applied to both arms
#   4  the step-down sweep over Delta, and the selected set at q = 0.1
#   ?  the same pipeline repeated over many datasets
#
# Run with --json <path> to dump the numbers the documentation page renders.
#
#   cd src/fdr && python PS_Fdr_Data_Example.py
#   cd src/fdr && python PS_Fdr_Data_Example.py --reps 200 --json out.json

import argparse
import json
import sys

import numpy as np

from ps_fdr import (fdp_power, lasso_support_cv, ps_fdr, sam_normalize,
                    stability_selection)

# --- the truth being recovered ------------------------------------------
# Five features carry signal, in descending strength; the other twenty-five
# are noise.  Feature 5 is deliberately weak enough to be missable.
BETA_STAR = np.array([1.6, 1.2, 0.9, 0.6, 0.4] + [0.0] * 25)
SIGMA = 1.5          # outcome noise
RHO = 0.35           # AR(1) correlation between neighbouring features
N_SUBJECTS = 70


def make_data(rng, n=N_SUBJECTS, beta=BETA_STAR, sigma=SIGMA, rho=RHO):
    """One synthetic cohort: correlated features, a sparse linear outcome."""
    p = len(beta)
    idx = np.arange(p)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(cov)

    X = rng.standard_normal((n, p)) @ L.T
    y = X @ beta + rng.normal(0.0, sigma, size=n)
    return X, y


def run_once(seed, q_levels=(0.05, 0.1, 0.2), B=50, M=100, null_mode='fixed'):
    """One cohort end to end. Returns CV-lasso and PS-Fdr outcomes."""
    rng = np.random.default_rng(seed)
    X, y = make_data(rng)
    truth = BETA_STAR != 0

    cv_mask = lasso_support_cv(X, y, random_state=seed)
    cv_fdp, cv_power = fdp_power(np.flatnonzero(cv_mask), truth)

    res = ps_fdr(X, y, q=max(q_levels), B=B, M=M, seed=seed,
                 null_mode=null_mode)

    out = {'cv': {'n_sel': int(cv_mask.sum()), 'fdp': cv_fdp,
                  'power': cv_power, 'selected': np.flatnonzero(cv_mask)},
           'res': res, 'X': X, 'y': y, 'truth': truth, 'levels': {}}

    for q in q_levels:
        ok = [s for s in res['sweep'] if s['fdr'] <= q]
        best = max(ok, key=lambda s: s['n_pos']) if ok else None
        sel = np.array(best['selected'] if best else [], dtype=int)
        fdp, power = fdp_power(sel, truth)
        out['levels'][q] = {'selected': sel, 'fdp': fdp, 'power': power,
                            'fdr_hat': best['fdr'] if best else float('nan'),
                            'n_sel': int(sel.size)}
    return out


def fmt_set(idx):
    return '{' + ', '.join(str(int(i) + 1) for i in sorted(idx)) + '}' if len(idx) else '{}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20260802)
    ap.add_argument('--B', type=int, default=50)
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--reps', type=int, default=0,
                    help='repeat the whole pipeline this many times')
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--json', type=str, default=None)
    ap.add_argument('--viz', type=str, default=None,
                    help='dump the per-step figure data for the docs page')
    args = ap.parse_args()

    truth = BETA_STAR != 0
    dump = {'beta_star': BETA_STAR.tolist(), 'sigma': SIGMA, 'rho': RHO,
            'n': N_SUBJECTS, 'B': args.B, 'M': args.M, 'seed': args.seed}

    one = run_once(args.seed, B=args.B, M=args.M)
    res, X, y = one['res'], one['X'], one['y']
    Pi, order = res['Pi'], res['order']

    # ---- 0 ---------------------------------------------------------------
    print(f"\n=== 0. cohort =========================================")
    print(f"n = {X.shape[0]} subjects, p = {X.shape[1]} features, "
          f"{int(truth.sum())} real")
    print(f"beta* = {np.array2string(BETA_STAR, precision=2)}")
    print(f"sigma = {SIGMA}, AR(1) rho = {RHO}")
    print(f"signal-to-noise (var explained) = "
          f"{np.var(X @ BETA_STAR) / np.var(y):.3f}")
    print(f"\nCV-tuned lasso on the full data selects {one['cv']['n_sel']}: "
          f"{fmt_set(one['cv']['selected'])}")
    print(f"  FDP = {one['cv']['fdp']:.3f}   power = {one['cv']['power']:.3f}")
    dump['cv'] = {'selected': [int(i) + 1 for i in one['cv']['selected']],
                  'n_sel': one['cv']['n_sel'], 'fdp': one['cv']['fdp'],
                  'power': one['cv']['power']}
    dump['snr'] = float(np.var(X @ BETA_STAR) / np.var(y))

    # ---- 1 ---------------------------------------------------------------
    print(f"\n=== 1. stability selection, B = {args.B} ================")
    print(f"{'feature':>8} {'true beta':>10} {'Pi_j':>8} {'hits':>6}")
    for j in np.argsort(-Pi):
        print(f"{j + 1:>8} {BETA_STAR[j]:>10.2f} {Pi[j]:>8.2f} "
              f"{int(round(Pi[j] * args.B)):>6}")
    print(f"\nbootstrap support sizes: min {res['counts'].min()}, "
          f"median {np.median(res['counts']):.1f}, max {res['counts'].max()}")
    print(f"k fixed on the null arm = {res['k']}")
    dump['Pi'] = [{'j': int(j) + 1, 'beta': float(BETA_STAR[j]),
                   'pi': float(Pi[j]), 'real': bool(truth[j])}
                  for j in np.argsort(-Pi)]
    dump['counts'] = res['counts'].tolist()
    dump['k'] = int(res['k'])

    # ---- 2, 3 ------------------------------------------------------------
    print(f"\n=== 2-3. null arm (M = {args.M}) and normalization ======")
    print(f"{'rank':>5} {'var':>4} {'Pi_(j)':>8} {'Pibar_(j)':>10} "
          f"{'Z_(j)':>8} {'Zbar_(j)':>9} {'gap':>8}")
    rows = []
    for r in range(len(Pi) - 1, -1, -1):       # print highest rank first
        j = order[r]
        row = {'rank': r + 1, 'j': int(j) + 1, 'pi': float(res['Z'][r]),
               'pi_raw': float(Pi[j]), 'pi_bar': float(res['Pi_bar'][r]),
               'z': float(res['Z'][r]), 'z_bar': float(res['Z_bar'][r]),
               'gap': float(res['Z'][r] - res['Z_bar'][r]),
               'real': bool(truth[j])}
        rows.append(row)
        print(f"{r + 1:>5} {j + 1:>4} {Pi[j]:>8.2f} {res['Pi_bar'][r]:>10.3f} "
              f"{res['Z'][r]:>8.2f} {res['Z_bar'][r]:>9.2f} "
              f"{res['Z'][r] - res['Z_bar'][r]:>8.2f}")
    dump['ordered'] = rows

    # ---- 4 ---------------------------------------------------------------
    print(f"\n=== 4. step-down sweep over Delta ======================")
    print(f"{'Delta':>8} {'Z(Delta)':>9} {'N+':>4} {'e0':>7} {'Fdr':>7}  set")
    sweep_rows = []
    seen = set()
    for s in res['sweep']:
        key = s['n_pos']
        if key in seen:
            continue
        seen.add(key)
        sel = np.array(s['selected'], dtype=int)
        fdp, power = fdp_power(sel, truth)
        print(f"{s['delta']:>8.2f} {s['cutoff']:>9.2f} {s['n_pos']:>4} "
              f"{s['e0']:>7.2f} {s['fdr']:>7.3f}  {fmt_set(sel)}")
        sweep_rows.append({'delta': s['delta'], 'cutoff': s['cutoff'],
                           'n_pos': s['n_pos'], 'e0': s['e0'],
                           'fdr': s['fdr'], 'fdp': fdp, 'power': power,
                           'selected': [int(i) + 1 for i in sorted(sel)]})
    dump['sweep'] = sweep_rows

    print()
    for q, lv in one['levels'].items():
        print(f"q = {q:<5} -> selects {lv['n_sel']} {fmt_set(lv['selected'])}, "
              f"Fdr-hat = {lv['fdr_hat']:.3f}, "
              f"FDP = {lv['fdp']:.3f}, power = {lv['power']:.3f}")
    dump['levels'] = {str(q): {'selected': [int(i) + 1 for i in sorted(lv['selected'])],
                               'n_sel': lv['n_sel'], 'fdr_hat': lv['fdr_hat'],
                               'fdp': lv['fdp'], 'power': lv['power']}
                      for q, lv in one['levels'].items()}

    # ---- repeated runs ---------------------------------------------------
    if args.reps:
        from joblib import Parallel, delayed

        print(f"\n=== ?. {args.reps} fresh cohorts ========================")
        seeds = [args.seed + 1000 + i for i in range(args.reps)]

        for mode in ('fixed', 'cv'):
            runs = Parallel(n_jobs=args.jobs, verbose=0)(
                delayed(run_once)(s, B=args.B, M=args.M, null_mode=mode)
                for s in seeds)

            label = ('permuted count fixed to k (as published)' if mode == 'fixed'
                     else 'permuted count re-chosen by CV (the invalid variant)')
            print(f"\n  null arm: {label}")
            print(f"    {'target q':>10} {'mean FDP':>9} {'power':>7} "
                  f"{'mean |S|':>9} {'FDP<=q':>7}")
            rep_rows = []
            for q in sorted(runs[0]['levels']):
                fdps = np.array([r['levels'][q]['fdp'] for r in runs])
                pw = np.array([r['levels'][q]['power'] for r in runs])
                ns = np.array([r['levels'][q]['n_sel'] for r in runs])
                print(f"    {q:>10} {fdps.mean():>9.3f} {pw.mean():>7.3f} "
                      f"{ns.mean():>9.2f} {np.mean(fdps <= q):>7.3f}")
                rep_rows.append({'q': q, 'fdp': float(fdps.mean()),
                                 'power': float(pw.mean()),
                                 'n_sel': float(ns.mean()),
                                 'hit': float(np.mean(fdps <= q))})
            dump[f'reps_{mode}'] = rep_rows

            if mode == 'fixed':
                cv_fdp = np.array([r['cv']['fdp'] for r in runs])
                cv_pw = np.array([r['cv']['power'] for r in runs])
                cv_n = np.array([r['cv']['n_sel'] for r in runs])
                print(f"    {'CV lasso':>10} {cv_fdp.mean():>9.3f} "
                      f"{cv_pw.mean():>7.3f} {cv_n.mean():>9.2f} "
                      f"{np.mean(cv_fdp <= 0.1):>7.3f}")
                dump['reps_cv_lasso'] = {'fdp': float(cv_fdp.mean()),
                                         'power': float(cv_pw.mean()),
                                         'n_sel': float(cv_n.mean()),
                                         'hit': float(np.mean(cv_fdp <= 0.1))}
        dump['reps'] = args.reps

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(dump, fh, indent=1)
        print(f"\nwrote {args.json}", file=sys.stderr)

    if args.viz:
        # One figure per step, all from this same run.  Column order is the
        # display order used throughout the page: most stable feature first.
        col = np.argsort(-Pi, kind='stable')
        viz = {
            'features': [int(j) + 1 for j in col],
            'real': [bool(truth[j]) for j in col],
            # step 1: the B x p selection grid the frequencies are counted from
            'masks': [''.join('1' if m else '0' for m in res['masks'][b, col])
                      for b in range(args.B)],
            'pi': [round(float(Pi[j]), 4) for j in col],
            # step 2: every permutation's ordered frequency curve, high rank first
            'null': [[int(round(v * args.B)) for v in row[::-1]]
                     for row in res['Pi_null']],
            'null_mean': [round(float(v), 4) for v in res['Pi_bar'][::-1]],
            'B': args.B, 'M': args.M,
            # step 3 is a closed-form map, recomputed in the page from pi
            # step 4: the ordered pairs the step-down rule compares
            'z': [round(float(v), 4) for v in res['Z'][::-1]],
            'z_bar': [round(float(v), 4) for v in res['Z_bar'][::-1]],
            'cutoff': [{'n_pos': s['n_pos'], 'fdr': round(s['fdr'], 4),
                        'cutoff': round(s['cutoff'], 4),
                        'e0': round(s['e0'], 3)}
                       for s in sweep_rows],
        }
        with open(args.viz, 'w') as fh:
            json.dump(viz, fh)
        print(f"wrote {args.viz}", file=sys.stderr)


if __name__ == '__main__':
    main()
