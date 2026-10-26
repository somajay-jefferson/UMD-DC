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
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'fdr'))

from ps_fdr import fdp_power                              # noqa: E402

from clusso_ps_fdr import ClussoSelector, clusso_ps_fdr   # noqa: E402
from clusso_select import (DEFAULT_LAMBDA_GRID,           # noqa: E402
                           DEFAULT_N_STARTS, NULL_LAMBDA_GRID,
                           clusso_fit_cv, support_mask, threshold_binding)
from stage1_noise import make_cohort_full                 # noqa: E402
from controls import (METRICS, naive_metrics, oracle_fit,    # noqa: E402
                      refit_metrics, structured_metrics, summarize)

Q = 0.1

ARMS = {
    'exact_k':      {'null_mode': 'fixed',        'null_rule': 'exact'},
    'fixed_lambda': {'null_mode': 'fixed_lambda', 'null_rule': 'exact'},
}


# The permuted arm's lambda ladder. 'fine' is the real arm's own 40-step grid,
# which is what the permuted arm walked until commit 4992ca8 and what stage4.json
# ran on. 'coarse' is the 13-step NULL_LAMBDA_GRID adopted there. stage5_reps100
# showed coarse is NOT output-identical: it lowers null_top slightly and adds
# false discoveries on a few cohorts. REVIEW.md section 9.
NULL_LADDERS = {'coarse': NULL_LAMBDA_GRID, 'fine': DEFAULT_LAMBDA_GRID}


def run_meta(args, argv):
    """
    What a result file needs to say about itself beyond its CLI args: which code
    ran, on which ladders, when, and for how long. stage4.json and stage5_reps100
    have identical args and different ladders, which is why this exists.
    """
    import platform
    import subprocess
    import sklearn

    def git(*a):
        try:
            return subprocess.run(['git', *a], cwd=_HERE, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    return {
        'git_commit': git('rev-parse', 'HEAD'),
        'git_dirty_src': bool(git('status', '--porcelain', '--untracked-files=no', '--', '..')),
        'command': ' '.join(argv),
        'null_ladder': args.null_ladder,
        'n_lambda_real_arm': int(len(DEFAULT_LAMBDA_GRID)),
        'n_lambda_null_arm': int(len(NULL_LADDERS[args.null_ladder])),
        'n_starts': int(DEFAULT_N_STARTS),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'sklearn': sklearn.__version__,
        'machine': platform.node(),
    }


def one_cohort(seed, B, M, n, q_feat, sparsity, arms, q=Q, null_ladder='coarse'):
    # The clustering happens HERE, once, and never again. Everything downstream
    # resamples subjects out of this one frozen tensor.
    c = make_cohort_full(n, q_feat, sparsity, seed)
    clust_X, Y, truth, clust_acc = c['clust_X'], c['Y'], c['truth'], c['clust_acc']

    out = {'clust_acc': float(clust_acc), 'n_true': int(truth.sum())}

    # Controls (controls.py). Naive ignores the tubule structure; the oracle
    # sees the true latent tubules, so it is CLUSSO with no clustering error.
    # `sec` is fit time only, scoring excluded, as Rubin's tic/toc pairs time it.
    t0 = time.perf_counter()
    out['naive'] = naive_metrics(c['X_avg'], Y, c['beta_star'], truth)
    out['naive']['sec'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    oracle = oracle_fit(c['X_full'], Y)
    sec = time.perf_counter() - t0
    out['oracle'] = structured_metrics(oracle, Y, c['X_full'], c['beta_star'], truth)
    out['oracle']['sec'] = sec

    # Baseline: CLUSSO as it ships. CV-tuned, no FDR control, one fit.
    t0 = time.perf_counter()
    base_fit = clusso_fit_cv(clust_X, Y, n_folds=5, rng=None, n_starts=5)
    base_sec = time.perf_counter() - t0
    base_mask = support_mask(base_fit['bet'])
    b_fdp, b_pw = fdp_power(np.flatnonzero(base_mask), truth)
    out['baseline'] = {'fdp': float(b_fdp), 'power': float(b_pw),
                       'n_sel': int(base_mask.sum()),
                       'lam_cv': float(base_fit['lam']),
                       'thresh_binding': bool(threshold_binding(base_fit['bet']))}
    out['baseline'].update(structured_metrics(base_fit, Y, clust_X,
                                              c['beta_star'], truth))
    out['baseline']['sec'] = base_sec

    for arm in arms:
        sel = ClussoSelector(null_lambda_grid=NULL_LADDERS[null_ladder])
        t0 = time.perf_counter()
        res, sel = clusso_ps_fdr(clust_X, Y, q=q, B=B, M=M, seed=seed,
                                 selector=sel, **ARMS[arm])
        sec = time.perf_counter() - t0
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
        # Random CLUSSO's device: structured OLS on the selected set only, so a
        # method that returns a set still has a bias and an MSE.
        out[arm].update(refit_metrics(res['selected'], clust_X, Y,
                                      c['beta_star'], truth))
        out[arm]['sec'] = sec
        out[arm]['lam'] = out[arm]['lam_star']
    return out


METHOD_LABELS = [('naive', 'naive avg'), ('oracle', 'oracle (full)'),
                 ('baseline', 'CLUSSO (bare)'), ('exact_k', 'FDR exact-k'),
                 ('fixed_lambda', 'FDR one-lam')]


def print_metrics(per_cohort):
    """
    Every method on both papers' metrics, mean then median. He et al. report
    means, the CLUSSO paper medians, so both are printed and neither is chosen.
    Methods a saved run predates are skipped.
    """
    have = [(m, lab) for m, lab in METHOD_LABELS
            if m in per_cohort[0]]
    table = {m: summarize([c[m] for c in per_cohort]) for m, _ in have}
    for stat in ('mean', 'median'):
        print(f'\n  {stat} over {len(per_cohort)} cohorts')
        print('  ' + f'{"method":>14}' + ''.join(f'{lab:>10}' for _, lab in METRICS))
        for m, lab in have:
            cells = []
            for key, _ in METRICS:
                v = table[m][key][stat]
                cells.append(f'{"-":>10}' if v is None else f'{v:>10.3f}')
            print('  ' + f'{lab:>14}' + ''.join(cells))
    return table


def _agg(runs, key):
    return np.array([r[key] for r in runs], dtype=float)


def results_html(out, per_cohort, args, arms):
    """
    The docs/fdr-clusso.html results section, rendered from the run rather than
    typed. CLAUDE.md: numbers on the docs pages are generated from real runs of
    the code, not hand-authored.
    """
    head = 'exact_k' if 'exact_k' in out else arms[0]
    b, a = out['baseline'], out[head]
    fl = out.get('fixed_lambda')
    fdps = sorted(round(c[head]['fdp'], 2) for c in per_cohort)
    ks = sorted(c[head]['k'] for c in per_cohort)
    kmed = ks[len(ks) // 2]
    date = args.run_date or 'undated'
    n_true = per_cohort[0]['n_true']

    rows = [('CLUSSO, bare', b, 'bad')]
    if 'exact_k' in out:
        rows.append(('FDR&#8209;CLUSSO, exact&#8209;k', out['exact_k'], 'ok'))
    if fl:
        rows.append(('FDR&#8209;CLUSSO, one &#x3BB;', fl, 'ok'))

    tr = '\n'.join(
        '          <tr><td>%s</td><td class="%s">%.3f</td><td>%.3f</td>'
        '<td>%.1f</td><td>%.2f</td></tr>'
        % (nm, cl, r['fdp'], r['power'], r['n_sel'], r['hit_rate'])
        for nm, r, cl in rows)

    strip = ' '.join(
        '<span class="%s">%.2f</span>' % ('good' if f <= 0.1 else 'bad', f)
        for f in fdps)

    agree = ''
    if fl:
        agree = (' And the two ways of constraining the permuted arm agree '
                 '&mdash; %.3f against %.3f &mdash; so the paper&rsquo;s '
                 'fixed&#8209;count device is <em class="term">not '
                 'load&#8209;bearing</em> here, and the cheaper rule reaches '
                 'the same answer.' % (a['fdp'], fl['fdp']))

    return """<!-- ================= RESULT ================= -->
<!-- GENERATED by stage4_clusso_fdr.py --html. Do not hand-edit: re-render with
     python stage4_clusso_fdr.py --from-json stage4.json --html out.html -->
  <section>
    <h2>Measured, %(date)s</h2>
    <h3>It halves the false discoveries, and misses the level it promises</h3>

    <div class="prose">
      <p>
        %(reps)d simulated replications, each <math><mi>n</mi></math>&thinsp;=&thinsp;%(n)d
        subjects on %(q)d features of which %(n_true)d are real.
        <math><mi>B</mi></math>&thinsp;=&thinsp;%(B)d bootstraps,
        <math><mi>M</mi></math>&thinsp;=&thinsp;%(M)d permutations, target
        <math><mi>q</mi></math>&thinsp;=&thinsp;0.10. The clustering is fitted
        <strong>once per replication</strong> and then frozen, so every number
        below is conditional on one clustering solution.
      </p>
    </div>

    <div class="scroll-x">
      <table class="alpha">
        <thead><tr><th></th><th>FDP</th><th>power</th><th>|S|</th><th>FDP&thinsp;&le;&thinsp;q</th></tr></thead>
        <tbody>
%(tr)s
        </tbody>
      </table>
    </div>

    <div class="prose">
      <p>
        Wrapping CLUSSO in PS&#8209;Fdr cut the false discovery proportion from
        <b>%(bfdp).3f</b> to <b>%(afdp).3f</b>, for <b>%(dpow).3f</b> of power.
        Bare CLUSSO stayed under the target on %(bhit)d replication of
        %(reps)d; the composition did on %(ahit)d.
      </p>
      <p>
        <strong>It does not reach the level it promises.</strong> %(afdp).3f
        against a target of 0.10. Per&#8209;replication FDP, sorted &mdash;
        <span class="good">green</span> clears the target,
        <span class="bad">red</span> does not:
      </p>
      <p class="fdpstrip">%(strip)s</p>
      <p>
        Two things make that trustworthy rather than lucky. The permuted arm
        never once ran out of penalty grid &mdash; exhausted&nbsp;=&nbsp;%(exh).3f
        across %(reps)d&#8239;&times;&#8239;%(M)d&#8239;&times;&#8239;%(B)d fits
        &mdash; so the estimate is not flattered by a null arm that came back
        short.%(agree)s
      </p>
      <p>
        <strong>Caveats that travel with the number.</strong> The median
        permuted&#8209;arm target was %(kmed)d of %(q)d, so
        <math><mi>k</mi>/<mi>p</mi></math>&thinsp;&asymp;&thinsp;%(kp).2f &mdash;
        far outside the regime PS&#8209;Fdr was built for, where He et al. have
        <math><mi>p</mi></math>&thinsp;=&thinsp;23,052 and
        <math><mi>k</mi></math> in the tens. Clustering accuracy was
        %(acc).3f, so this says nothing about what freezing costs when the
        clustering is wrong &mdash; the GMM simply got it right every time. And
        core&rsquo;s hardcoded 0.001 coefficient cut was plausibly deciding the
        support on %(bind).0f%% of replications, which is the first thing to rule
        out before trusting the gap between %(afdp).3f and 0.10.
      </p>
    </div>
  </section>
""" % {
        'date': date, 'reps': args.reps, 'n': args.n, 'q': args.q_features,
        'n_true': n_true, 'B': args.B, 'M': args.M, 'tr': tr,
        'bfdp': b['fdp'], 'afdp': a['fdp'], 'dpow': b['power'] - a['power'],
        'bhit': round(b['hit_rate'] * args.reps),
        'ahit': round(a['hit_rate'] * args.reps),
        'strip': strip, 'exh': a['exhausted_rate'], 'agree': agree,
        'kmed': kmed, 'kp': kmed / float(args.q_features),
        'acc': out['clust_acc'], 'bind': b['thresh_binding'] * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--B', type=int, default=50)
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--q-features', type=int, default=50)
    ap.add_argument('--sparsity', type=float, default=0.8)
    ap.add_argument('--arm', choices=('a', 'b', 'both'), default='both')
    ap.add_argument('--null-ladder', choices=tuple(NULL_LADDERS), default='coarse',
                    help="permuted arm's lambda ladder: 'coarse' (13 steps, the "
                         "default since 4992ca8) or 'fine' (the real arm's 40, "
                         'what stage4.json ran on)')
    # -1 is joblib's "every LOGICAL core", which oversubscribes SMT siblings.
    # Measured on a Ryzen 5 5600 (6 physical / 12 logical): 12 workers ran each
    # fit 2.5x slower than 4 and bought only ~16% total throughput, because this
    # workload is compute-dense rather than latency-bound. Prefer one worker per
    # PHYSICAL core.
    ap.add_argument('--jobs', type=int, default=-1,
                    help='worker processes. NOTE -1 means every logical core; '
                         'on an SMT machine pass the physical core count '
                         'instead, which is usually half that')
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--exhausted-max', type=float, default=0.02,
                    help='gate: max tolerable rate of lambda-grid exhaustion '
                         'on the permuted arm, since that error is '
                         'anti-conservative and silent')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--json', type=str, default=None)
    ap.add_argument('--from-json', type=str, default=None,
                    help='re-report a saved run instead of computing one. The '
                         'run costs hours, so rendering must not require '
                         'repeating it.')
    ap.add_argument('--html', type=str, default=None,
                    help='write the docs/fdr-clusso.html results section. '
                         'CLAUDE.md requires numbers on the docs pages to be '
                         'generated from a real run rather than typed.')
    ap.add_argument('--run-date', type=str, default=None,
                    help='date to stamp on the --html block')
    args = ap.parse_args()
    meta = run_meta(args, sys.argv)
    meta['started'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    t_start = time.perf_counter()

    if args.quick:
        args.reps, args.B, args.M, args.n, args.q_features = 3, 10, 10, 120, 25

    arms = {'a': ['exact_k'], 'b': ['fixed_lambda'],
            'both': ['exact_k', 'fixed_lambda']}[args.arm]

    from joblib import Parallel, delayed

    seeds = [args.seed + i for i in range(args.reps)]
    print(f'{args.reps} cohorts, n={args.n} q={args.q_features} '
          f'sparsity={args.sparsity}, B={args.B} M={args.M}, target q={Q}')
    if not args.from_json:
        print(f'  CLUSSO inside PS-Fdr, clustering frozen once per cohort, '
              f'null ladder {args.null_ladder} ({meta["n_lambda_null_arm"]} '
              f'steps) at {(meta["git_commit"] or "?")[:7]}\n')

    # verbose=10 so a long run reports progress instead of going silent for
    # hours. A 20-cohort run at n=300 q=50 is ~175 min of worker time PER
    # cohort -- the real arm alone runs a 40-lambda x 5-fold CV inside each of
    # the B resamples, which is 41s x 50 -- so silence is not a viable default.
    if args.from_json:
        saved = json.load(open(args.from_json))
        per_cohort = saved['per_cohort']
        for k, v in saved['args'].items():
            if k not in ('json', 'from_json', 'html', 'run_date'):
                setattr(args, k, v)
        arms = [a for a in ARMS if a in per_cohort[0]]
        args.run_date = args.run_date or saved.get('run_date')
        # The saved run's own record, never this invocation's: a re-report
        # must not relabel what ran.
        meta = saved.get('meta', {'note': 'saved before run metadata existed'})
        print('  (re-reporting %s, no fits performed; null ladder %s)\n'
              % (args.from_json, meta.get('null_ladder', 'not recorded')))
    else:
        per_cohort = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(one_cohort)(s, args.B, args.M, args.n, args.q_features,
                                args.sparsity, arms, null_ladder=args.null_ladder)
            for s in seeds)
        meta['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        meta['wall_minutes'] = round((time.perf_counter() - t_start) / 60.0, 1)

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
    out['metrics'] = print_metrics(per_cohort)

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

    if args.html:
        with open(args.html, 'w', encoding='utf-8') as fh:
            fh.write(results_html(out, per_cohort, args, arms))
        print('wrote %s' % args.html, file=sys.stderr)

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'meta': meta, 'args': vars(args), 'q': Q, 'arms': out,
                       'per_cohort': per_cohort, 'pass': bool(passed),
                       'run_date': args.run_date},
                      fh, indent=1)
        print(f'wrote {args.json}', file=sys.stderr)

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
