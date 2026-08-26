# Verification for the FDR-CLUSSO selection rule.
#
# This repo has no test suite, so "it works" has been an assertion. These are
# the checks that could actually catch it being wrong, run with one command:
#
#     python check.py
#
# The valuable ones are the PARITY checks. clusso_select.py wraps code that
# already exists and is already trusted, so the question that matters is not
# "does it look right" but "does it reproduce what src/core/ does". Anywhere the
# answer is no, either the wrapper is broken or the difference is deliberate --
# and the deliberate ones are named here so they cannot hide.
#
# Deliberate differences, checked as such rather than swept aside:
#
#   Folds are built once per grid sweep, not rebuilt per lambda. lambda_CV_mse
#   rebuilds them on every call (SLasso_MSE.py:129), so core scores each lambda
#   against a different partition. Check 3 pins my CV loop against core's
#   scoring on a FIXED partition, which isolates the loop from that difference.
#
#   alpha_init/beta_init are pinned rather than drawn. That is the whole point,
#   so parity is checked at matched inits (check 1), not against core's draws.
#
# Exit code is 0 only if every check passes.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'core'))

from Mainfunction_albet import Mainfunction_albet          # noqa: E402
from SLasso_MSE import CV_make_folds, slasso_mse           # noqa: E402

from clusso_select import (DEFAULT_N_STARTS, _score_fold, alpha_starts,
                           clusso_fit, clusso_fit_cv,
                           clusso_fit_multistart, clusso_objective,
                           clusso_support_at_least_k, clusso_support_cv,
                           cv_mse, make_folds, support_mask)   # noqa: E402


RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail or ''))
    except AssertionError as e:
        RESULTS.append((False, name, str(e)))
    except Exception as e:                                  # noqa: BLE001
        RESULTS.append((False, name, f'{type(e).__name__}: {e}'))


def toy(seed=0, P=2, q=25, n=90, sigma=1.0):
    """A small problem with real signal, from an explicit generator."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(P, q, n))
    beta = np.zeros(q)
    beta[:5] = [1.6, 1.2, 0.9, 0.6, 0.4]
    alpha = np.array([0.7, 0.3]) if P == 2 else rng.normal(size=P)
    y = np.einsum('i,ijk,j->k', alpha, X, beta) + rng.normal(0, sigma, n)
    return X, y, beta != 0


# ---------------------------------------------------------------------------
# parity: does the wrapper reproduce src/core exactly?
# ---------------------------------------------------------------------------

def c1_fit_is_passthrough():
    """clusso_fit must be Mainfunction_albet and nothing else."""
    X, y, _ = toy()
    P, q = X.shape[0], X.shape[1]
    a0, b0 = np.ones(P), np.ones(q)

    for lam in (0.05, 0.5, 3.0):
        mine = clusso_fit(X, y, lam, alpha_init=a0, beta_init=b0)
        core = Mainfunction_albet(X, y, a0, b0, lam)
        assert np.array_equal(mine['alpha'], core['alpha']), f'alpha differs at lam={lam}'
        assert np.array_equal(mine['bet'], core['bet']), f'bet differs at lam={lam}'
    return 'identical alpha and bet at 3 lambdas'


def c2_score_fold_matches_slasso_mse():
    """
    _score_fold duplicates the scoring inside slasso_mse -- including its two
    easy-to-get-wrong centering conventions. This is the check that catches that
    copy drifting, and it is the single most likely place for a silent bug.
    """
    X, y, _ = toy(seed=3, n=100)
    P, q = X.shape[0], X.shape[1]
    a0, b0 = np.ones(P), np.ones(q)

    folds = make_folds(X.shape[2], 5)
    all_idx = np.arange(X.shape[2])
    worst = 0.0

    for k in range(5):
        te = folds[k]
        tr = np.setdiff1d(all_idx, te)
        lam = 0.4

        core = slasso_mse(X[:, :, tr], y[tr], X[:, :, te], y[te], a0, b0, lam)
        fit = clusso_fit(X[:, :, tr], y[tr], lam, alpha_init=a0, beta_init=b0)
        mine = _score_fold(X[:, :, te], y[te], fit['alpha'], fit['bet'])

        worst = max(worst, abs(core - mine))
        assert np.isclose(core, mine, rtol=0, atol=1e-12), (
            f'fold {k}: slasso_mse={core!r} but _score_fold={mine!r}')
    return f'agrees to {worst:.2e} across 5 folds'


def c3_cv_loop_matches_core_scoring():
    """
    My CV loop, on a FIXED partition, must equal the mean of core's slasso_mse
    over that same partition. Isolates the loop from the deliberate
    build-folds-once difference.
    """
    X, y, _ = toy(seed=5, n=95)
    P, q = X.shape[0], X.shape[1]
    a0, b0 = np.ones(P), np.ones(q)
    folds = make_folds(X.shape[2], 5)
    all_idx = np.arange(X.shape[2])

    for lam in (0.1, 1.0):
        manual = np.mean([
            slasso_mse(X[:, :, np.setdiff1d(all_idx, f)],
                       y[np.setdiff1d(all_idx, f)],
                       X[:, :, f], y[f], a0, b0, lam)
            for f in folds])
        mine = cv_mse(X, y, lam, folds, n_starts=1)
        assert np.isclose(manual, mine, rtol=0, atol=1e-12), (
            f'lam={lam}: manual={manual!r} mine={mine!r}')

        # The n_starts=1 path calls core's slasso_mse, so it never touches
        # _score_fold. Score the same folds through _score_fold as well, or the
        # multi-start path -- the only place _score_fold is used in anger --
        # goes unchecked. Mutation testing found this gap; without these lines a
        # broken _score_fold passes this check.
        via_score_fold = np.mean([
            _score_fold(X[:, :, f], y[f],
                        *(lambda fit: (fit['alpha'], fit['bet']))(
                            clusso_fit_multistart(
                                X[:, :, np.setdiff1d(all_idx, f)],
                                y[np.setdiff1d(all_idx, f)], lam, n_starts=1)))
            for f in folds])
        assert np.isclose(manual, via_score_fold, rtol=0, atol=1e-12), (
            f'lam={lam}: _score_fold path={via_score_fold!r} vs core={manual!r}')
    return 'CV mean matches core slasso_mse exactly, both scoring paths'


def c4_support_is_cores_nonzeros():
    """support_mask at tau=0 must be exactly core's own nonzeros, nothing more."""
    X, y, _ = toy(seed=7)
    for lam in (0.05, 0.8):
        bet = clusso_fit(X, y, lam)['bet']
        assert np.array_equal(support_mask(bet, tau=0.0), bet != 0.0), \
            f'support_mask disagrees with (bet != 0) at lam={lam}'
    return 'tau=0 is exactly (bet != 0)'


def c5_folds_match_core_convention():
    """
    make_folds must partition the way CV_make_folds does: disjoint, covering,
    first n_folds-1 of size floor(n/n_folds), last taking the remainder.
    """
    for n in (20, 37, 90, 103):
        np.random.seed(0)
        core = CV_make_folds(n)
        for rng in (None, np.random.default_rng(1)):
            mine = make_folds(n, 5, rng=rng)
            assert len(mine) == len(core) == 5, f'n={n}: fold count'
            assert sorted(np.concatenate(mine).tolist()) == list(range(n)), \
                f'n={n}: folds do not partition 0..n-1'
            assert [len(f) for f in mine] == [len(f) for f in core], (
                f'n={n}: sizes {[len(f) for f in mine]} vs core '
                f'{[len(f) for f in core]}')
    try:
        make_folds(3, 5)
        raise AssertionError('n < n_folds should raise, as core does')
    except ValueError:
        pass
    return 'sizes and partition match CV_make_folds for n in 20,37,90,103'


# ---------------------------------------------------------------------------
# the properties the composition depends on
# ---------------------------------------------------------------------------

def c6_purity_under_hostile_global_seeds():
    """
    The claim the whole module rests on. Different global seeds, and an
    unrelated draw in between, must not move the answer by even one feature.
    """
    X, y, _ = toy(seed=11)
    grid = np.geomspace(0.01, 20.0, 12)

    # Compare the FULL CV trace, not just the final support. A support can be
    # robust enough to survive a fold change by luck, which makes a
    # support-only comparison a test that passes for the wrong reason --
    # mutation testing caught exactly that here.
    # n_starts must match the shipping default. Code review caught an earlier
    # version of this check calling clusso_fit_cv bare, which defaults to
    # n_starts=1 -- so the multi-start path that actually ships, and the
    # _score_fold scoring it routes through, went unperturbed.
    np.random.seed(1)
    fa = clusso_fit_cv(X, y, lambda_grid=grid, n_starts=DEFAULT_N_STARTS)
    a = support_mask(fa['bet'])

    np.random.seed(9999)
    _ = np.random.normal(size=17)          # unrelated draw, mid-stream
    fb = clusso_fit_cv(X, y, lambda_grid=grid, n_starts=DEFAULT_N_STARTS)
    b = support_mask(fb['bet'])

    assert fa['lam'] == fb['lam'], (
        f'CV picked different lambdas under different global seeds: '
        f'{fa["lam"]} vs {fb["lam"]}')
    assert np.array_equal(fa['mse'], fb['mse']),         'the CV error curve itself moved under a different global seed'
    assert [f.tolist() for f in fa['folds']] == [f.tolist() for f in fb['folds']],         'fold assignment moved under a different global seed'

    np.random.seed(1)
    c = clusso_support_at_least_k(X, y, 8, lambda_grid=grid)[0]
    np.random.seed(424242)
    d = clusso_support_at_least_k(X, y, 8, lambda_grid=grid)[0]

    assert np.array_equal(a, b), 'clusso_support_cv moved under a different global seed'
    assert np.array_equal(c, d), 'at_least_k moved under a different global seed'
    return (f'lambda, CV curve, folds and support all stable '
            f'(k={int(a.sum())}, {int(c.sum())})')


def c7_determinism_on_repeat():
    """Same call, ten times, one answer."""
    X, y, _ = toy(seed=13)
    grid = np.geomspace(0.01, 20.0, 10)
    seen = {tuple(np.flatnonzero(clusso_support_cv(X, y, lambda_grid=grid)))
            for _ in range(10)}
    assert len(seen) == 1, f'{len(seen)} distinct supports across 10 identical calls'

    seen_ms = {tuple(np.round(clusso_fit_multistart(X, y, 0.3, n_starts=5)['bet'], 12))
               for _ in range(5)}
    assert len(seen_ms) == 1, 'multistart is not deterministic'
    return '10 identical calls, 1 answer; multistart stable'


def c8_at_least_k_contract():
    """
    Two halves, and the second is the one that matters.

    Weak half: never a silent shortfall -- either the support reaches k, or
    exhausted is True.

    Strong half: exhausted must be EARNED. Code review showed the weak half
    alone is satisfied by a rule that returns an empty support with
    exhausted=True for every k -- which would make the permuted arm select
    nothing, e0 zero, and Fdr-hat zero. That is precisely the anti-conservative
    failure this check exists to prevent, and it passed. So: for any k the grid
    can actually reach, exhausted must be False and the support must reach k.
    """
    X, y, _ = toy(seed=17, q=25)
    grid = np.geomspace(0.01, 20.0, 20)

    # what the grid can actually deliver, measured rather than assumed
    k_max = max(int(support_mask(clusso_fit_multistart(X, y, lam, n_starts=1)['bet']).sum())
                for lam in grid)
    assert k_max >= 5, f'fixture too weak to test with: grid tops out at {k_max}'

    for k in (1, 5, 12, 20, 25, 40):
        mask, info = clusso_support_at_least_k(X, y, k, lambda_grid=grid,
                                               n_starts=1)
        got = int(mask.sum())
        assert info['k_realised'] == got, f'k={k}: k_realised disagrees with the mask'
        assert got >= k or info['exhausted'], (
            f'k={k}: got {got} and exhausted is False -- silent shortfall')
        if k <= k_max:
            assert got >= k, f'k={k} is reachable (grid reaches {k_max}) but got {got}'
            assert not info['exhausted'], (
                f'k={k} is reachable but reported exhausted -- '
                f'a null arm that gives up here under-controls FDR')
        if k > X.shape[1]:
            assert info['exhausted'], f'k={k} exceeds q; must report exhausted'
    return f'reaches every k the grid supports (k_max={k_max}); flags the rest'


def c9_multistart_never_worse():
    """
    More starts must never return a worse objective than fewer. If it does, the
    start grid is not nested and the "5 is enough" claim is unsupported.
    """
    X, y, _ = toy(seed=19)

    for lam in (0.05, 0.5, 2.0):
        objs = []
        for ns in range(1, 10):
            f = clusso_fit_multistart(X, y, lam, n_starts=ns)
            objs.append(clusso_objective(X, y, f['alpha'], f['bet'], lam))
        for a, b in zip(objs, objs[1:]):
            assert b <= a + 1e-9, (
                f'lam={lam}: objective rose with more starts -- '
                f'{objs} over n_starts=1..9')
    return 'objective non-increasing over n_starts 1..9, at 3 lambdas'


def c11_alpha_starts_nested_and_exact():
    """
    alpha_starts was never tested. Code review showed the whole suite passes
    with it gutted to a single start -- silently reverting to the rule this
    branch measures at +2.95 objective gap.

    Two properties everything else leans on: exactly n_starts arrays, and sets
    nested in n_starts. Nesting is what makes 'more starts is never worse'
    (c9) true rather than merely usually true.
    """
    for P in (2, 3):
        for n in range(1, 16):
            got = alpha_starts(P, n)
            assert len(got) == n, f'P={P}, n={n}: got {len(got)} starts'
            assert all(np.any(np.abs(v) > 0) for v in got),                 f'P={P}, n={n}: a start is all zeros'
        for n in range(1, 15):
            small, big = alpha_starts(P, n), alpha_starts(P, n + 1)
            assert all(np.array_equal(a, b) for a, b in zip(small, big)),                 f'P={P}: start set at n={n} is not a prefix of n={n + 1}'

    assert len(alpha_starts(2, 1)) == 1 and np.array_equal(
        alpha_starts(2, 1)[0], np.ones(2)), 'n_starts=1 must be ones(P)'
    return 'exact length and nesting for P=2,3 over n_starts 1..15'


def c10_recovers_signal_and_rejects_noise():
    """
    Sanity, and a negative control. With signal it must find the real features;
    on pure noise it must select nothing. A suite that only ever runs on signal
    cannot tell you the method is capable of saying "nothing here".
    """
    grid = np.geomspace(0.01, 20.0, 15)

    X, y, truth = toy(seed=23, n=120, sigma=0.5)
    m = clusso_support_cv(X, y, lambda_grid=grid)
    found = int(m[truth].sum())
    assert found >= 4, f'found only {found}/5 real features with strong signal'

    rng = np.random.default_rng(29)
    Xn = rng.normal(size=(2, 25, 120))
    yn = rng.normal(size=120)                      # no relationship whatsoever
    kn = int(clusso_support_cv(Xn, yn, lambda_grid=grid).sum())
    assert kn <= 2, f'selected {kn} features from pure noise'
    return f'{found}/5 real found; {kn} selected from pure noise'


def main():
    check('parity   clusso_fit is Mainfunction_albet', c1_fit_is_passthrough)
    check('parity   _score_fold == slasso_mse', c2_score_fold_matches_slasso_mse)
    check('parity   CV loop == core scoring', c3_cv_loop_matches_core_scoring)
    check('parity   support_mask == core nonzeros', c4_support_is_cores_nonzeros)
    check('parity   make_folds == CV_make_folds convention',
          c5_folds_match_core_convention)
    check('purity   hostile global seeds', c6_purity_under_hostile_global_seeds)
    check('purity   determinism on repeat', c7_determinism_on_repeat)
    check('contract at_least_k reaches k or flags', c8_at_least_k_contract)
    check('contract objective non-increasing in n_starts', c9_multistart_never_worse)
    check('contract alpha_starts nested and exact', c11_alpha_starts_nested_and_exact)
    check('sanity   finds signal, rejects pure noise',
          c10_recovers_signal_and_rejects_noise)

    print()
    for ok, name, detail in RESULTS:
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
        if detail:
            print(f'          {detail}')
    n_ok = sum(ok for ok, _, _ in RESULTS)
    print(f'\n  {n_ok}/{len(RESULTS)}')
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
