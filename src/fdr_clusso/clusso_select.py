# CLUSSO as a pure, deterministic selection rule.
#
# PS-Fdr (src/fdr/ps_fdr.py) needs one thing from an estimator: given a design
# and an outcome, hand back the set of variables it kept. This module supplies
# that for CLUSSO, in the two forms PS-Fdr asks for -- a CV-tuned rule for the
# real-data arm, and a size-constrained rule for the permuted arm.
#
# Two properties are load-bearing and are the reason this file exists rather
# than the call sites in src/core/ being reused directly:
#
#   Pure. Nothing here touches the global numpy random stream. `Mainfunction_albet`
#   is already a pure deterministic function of its arguments -- sklearn's Lasso
#   is cyclic, lstsq is deterministic -- so CLUSSO's seed-dependence enters
#   entirely through `alpha_init`, which callers draw themselves
#   (CLUSSO_Functions_Project1_6_16_23.py:270). We pin it instead. The only
#   impure dependency in the fitting path is `CV_make_folds`, which draws from
#   the global stream by design (SLasso_MSE.py:72-74); `make_folds` below
#   replaces it with an explicit rng.
#
#   Frozen clustering. The (P, q, n) design tensor is built once, outside any
#   resampling loop, and resampled along axis 2. PS-Fdr assumes a fixed design;
#   CLUSSO normally manufactures one by clustering. Freezing it is what makes
#   the two compose at all, and it means ANY false discovery guarantee obtained
#   this way is conditional on that one clustering solution. That is a real
#   caveat, not a technicality -- clustering accuracy dominates CLUSSO's error.
#
# src/core/ is not modified. `Mainfunction_albet` and `slasso_mse` are called
# directly; only the fold logic is reimplemented.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'core'))

from Mainfunction_albet import Mainfunction_albet   # noqa: E402
from SLasso_MSE import slasso_mse                   # noqa: E402


# Spans sparse to nearly dense on the real-data shape. The top end must reach a
# support of one; the bottom end must approach q. stage2_path.py checks both.
DEFAULT_LAMBDA_GRID = np.geomspace(0.01, 20.0, 40)

# Hardcoded inside Mainfunction_albet at line 124, applied after beta is
# L1-normalised at line 104. Repeated here only for the threshold_binding
# diagnostic -- we cannot change it without editing src/core/.
CORE_BETA_THRESH = 0.001

# Number of deterministic starts the selection rules use by default.
#
# Measured in stage1_noise.py (n=300, q=50): a single pinned start lands in
# materially worse optima -- median objective gap +2.95 against best-of-100
# random starts. The five-point arc grid closes that to +0.0027, and going to
# nine buys nothing further. So one start is deterministic but not good enough,
# and five is the cheapest setting that is both.
DEFAULT_N_STARTS = 5


# ---------------------------------------------------------------------------
# deterministic fitting
# ---------------------------------------------------------------------------

def alpha_starts(P, n_starts=1):
    """
    Deterministic replacement for CLUSSO's random restarts.

    `Mainfunction_albet` normalises alpha to ``sum|alpha| == 1`` and forces the
    largest-magnitude entry positive (lines 79-81). At P=2 that collapses the
    entire initialisation space to two one-parameter arcs, ``(t, +(1-t))`` and
    ``(t, -(1-t))`` for t in [0, 1] -- so drawing `alpha_init` from a normal is
    really just sampling an interval at random. A grid over that interval covers
    the same space by construction and is reproducible.

    ``n_starts=1`` returns ``[ones(P)]``, which normalises to the uniform
    weighting -- the natural neutral start.

    Returns a list of (P,) arrays.
    """
    P = int(P)
    if n_starts <= 1:
        return [np.ones(P)]

    starts = [np.ones(P)]
    n_t = int(np.ceil((n_starts - 1) / 2.0))
    for sign in (1.0, -1.0):
        for t in np.linspace(0.15, 0.85, n_t):
            v = np.full(P, (1.0 - t) / max(P - 1, 1))
            v[0] = t
            v[1:] *= sign
            # skip anything the normalisation would collapse onto an existing start
            if any(np.allclose(v / np.sum(np.abs(v)), s / np.sum(np.abs(s)))
                   for s in starts):
                continue
            starts.append(v)
            if len(starts) >= n_starts:
                return starts
    return starts


def clusso_fit(X, y, lam, alpha_init=None, beta_init=None):
    """
    One CLUSSO fit. Deterministic: same arguments in, same result out, always.

    X is (P, q, n) -- clusters, features, subjects. y is (n,).

    `beta_init` defaults to ones rather than being left to the caller. It is
    very nearly inert -- the lasso solve at Mainfunction_albet.py:100 overwrites
    it before it is ever read numerically -- but it is still read by the loop
    guard at line 89 and is ``bet0`` in the first convergence check at line 120,
    so a pathological value could stop the loop after one pass. Pinning it
    removes that possibility.

    Returns ``{'alpha': (P,), 'bet': (q,), 'lam': float}``. Note alpha comes back
    rescaled by the pre-normalisation L1 norm (line 125) while bet stays on the
    L1 scale -- they are on different scales. See `clusso_objective`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    P, q, _ = X.shape

    if alpha_init is None:
        alpha_init = np.ones(P)
    if beta_init is None:
        beta_init = np.ones(q)

    fit = Mainfunction_albet(X, y, alpha_init, beta_init, float(lam))
    return {'alpha': fit['alpha'], 'bet': fit['bet'], 'lam': float(lam)}


def clusso_objective(X, y, alpha, bet, lam):
    """
    Penalised objective, following CLUSSO_Data_Example.py:162-166 exactly: the
    design centred over the subject axis, the outcome mean-centred, alpha and
    bet used as returned rather than rescaled onto a common footing.

    Used only to choose among multiple starts, so what matters is that it
    matches the existing convention rather than that it is the tidiest way to
    write the model.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    Xc = X - X.mean(axis=2, keepdims=True)
    yc = y - y.mean()

    resid = yc - np.einsum('i,ijk,j->k', alpha, Xc, bet)
    return float(resid @ resid + float(lam) * np.sum(np.abs(bet)))


def clusso_fit_multistart(X, y, lam, n_starts=1):
    """
    Fit from each deterministic start and keep the lowest objective. Ties go to
    the earlier start, so the result is a pure function of (X, y, lam, n_starts).
    """
    if n_starts <= 1:
        return clusso_fit(X, y, lam)

    P = np.asarray(X).shape[0]
    best, best_obj = None, np.inf
    for a0 in alpha_starts(P, n_starts):
        fit = clusso_fit(X, y, lam, alpha_init=a0)
        obj = clusso_objective(X, y, fit['alpha'], fit['bet'], lam)
        if obj < best_obj:
            best, best_obj = fit, obj
    best['objective'] = best_obj
    return best


# ---------------------------------------------------------------------------
# cross-validation, with no global random state
# ---------------------------------------------------------------------------

def make_folds(n, n_folds=5, rng=None):
    """
    Partition subject indices into folds. Reimplements
    `SLasso_MSE.CV_make_folds`, which draws from the global numpy stream.

    ``rng=None`` gives contiguous folds and takes no random draws at all. That
    is not a compromise in the PS-Fdr setting: the bootstrap index arriving from
    `stability_selection` is already in random order, so contiguous folds over
    the resampled subjects are random folds. It makes the whole selection rule a
    pure function of (X, y).

    Pass an rng to shuffle explicitly -- stage 1 needs that to isolate fold
    noise from initialisation noise.

    Size convention matches the R original: the first ``n_folds - 1`` folds take
    ``floor(n / n_folds)`` each and the last takes the remainder. Raises for
    ``n < n_folds``, as the existing port does.
    """
    n = int(n)
    n_folds = int(n_folds)
    if n < n_folds:
        raise ValueError(
            f"need at least {n_folds} subjects for {n_folds}-fold CV, got {n}"
        )

    idx = np.arange(n)
    if rng is not None:
        idx = rng.permutation(idx)

    size = n // n_folds
    folds = [np.sort(idx[k * size:(k + 1) * size]) for k in range(n_folds - 1)]
    folds.append(np.sort(idx[(n_folds - 1) * size:]))
    return folds


def _score_fold(X_test, Y_test, alpha_hat, beta_hat):
    """
    Held-out MSE under `slasso_mse`'s conventions (SLasso_MSE.py:45-60): centre
    the test outcomes, and centre each test matrix by the test fold's own
    subject-wise mean -- not a mean carried over from training.

    Only used on the multi-start path, where we need to score a fit we already
    have. The single-start path calls core's `slasso_mse` directly.
    """
    Y_test = np.asarray(Y_test, dtype=float).ravel()
    n_test = X_test.shape[2]

    Y_c = Y_test - Y_test.mean()
    X_mean = X_test.mean(axis=2)

    sse = 0.0
    for i in range(n_test):
        Xi = X_test[:, :, i] - X_mean
        sse += (Y_c[i] - alpha_hat @ Xi @ beta_hat) ** 2
    return float(sse / n_test)


def cv_mse(X, y, lam, folds, n_starts=1):
    """
    Mean held-out MSE at one lambda, over pre-built folds.

    The folds are supplied rather than built here, which is the one deliberate
    departure from `lambda_CV_mse`: that function rebuilds folds on every call,
    so a grid sweep scores each lambda against a different partition and the
    resulting MSEs are not strictly comparable.
    """
    y = np.asarray(y, dtype=float).ravel()
    n_folds = len(folds)
    all_idx = np.arange(X.shape[2])

    out = np.zeros(n_folds)
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.setdiff1d(all_idx, test_idx)

        X_tr, y_tr = X[:, :, train_idx], y[train_idx]
        X_te, y_te = X[:, :, test_idx], y[test_idx]

        if n_starts <= 1:
            # core's own function, so the single-start path stays byte-for-byte
            # the behaviour the R port defines
            out[k] = slasso_mse(X_tr, y_tr, X_te, y_te,
                                np.ones(X.shape[0]), np.ones(X.shape[1]),
                                float(lam))
        else:
            fit = clusso_fit_multistart(X_tr, y_tr, lam, n_starts=n_starts)
            out[k] = _score_fold(X_te, y_te, fit['alpha'], fit['bet'])

    return float(out.mean())


def clusso_fit_cv(X, y, lambda_grid=None, n_folds=5, rng=None, n_starts=1):
    """
    CV-tuned CLUSSO. Folds are built once and shared across the whole grid, then
    the winner is refit on all subjects.

    First minimiser wins ties, matching CLUSSO_Data_Example.py:150.

    Returns ``{'alpha', 'bet', 'lam', 'mse', 'folds'}``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    grid = DEFAULT_LAMBDA_GRID if lambda_grid is None else np.asarray(lambda_grid,
                                                                      dtype=float)

    folds = make_folds(X.shape[2], n_folds=n_folds, rng=rng)
    mse = np.array([cv_mse(X, y, lam, folds, n_starts=n_starts) for lam in grid])

    lam = float(grid[int(np.argmin(mse))])
    fit = clusso_fit_multistart(X, y, lam, n_starts=n_starts)
    fit['mse'] = mse
    fit['folds'] = folds
    return fit


# ---------------------------------------------------------------------------
# support
# ---------------------------------------------------------------------------

def support_mask(bet, tau=0.0):
    """
    Which features the fit kept, as a (q,) bool.

    ``tau=0`` takes core's own nonzeros. That is the recommended default: bet is
    returned L1-normalised and already cut at 0.001, so anything we do here can
    only shrink the support further -- we cannot recover a coefficient core has
    already zeroed. There is no principled reason to be stricter by default, and
    tau=0 keeps this comparable to the TPR/FPR the CLUSSO paper reports
    (CLUSSO_Functions_Project1_6_16_23.py:253-256).

    ``tau>0`` drops coefficients at or below tau as well. Since bet is on the L1
    scale, tau reads as a fraction of total coefficient mass. Useful as a
    sensitivity check, not as a default.
    """
    bet = np.abs(np.asarray(bet, dtype=float).ravel())
    return bet > max(float(tau), 0.0)


def threshold_binding(bet, factor=2.0):
    """
    True when core's hardcoded 0.001 cut is plausibly deciding the support
    rather than the lasso being the thing that zeroed coefficients.

    Worth logging because the cut interacts badly with the L1 normalisation: the
    more features survive, the smaller every coefficient becomes, so the same
    absolute threshold bites harder exactly when the support is large. It goes
    from inert to marginal somewhere around a support of thirty.
    """
    nz = np.abs(np.asarray(bet, dtype=float).ravel())
    nz = nz[nz > 0]
    if nz.size == 0:
        return False
    return bool(nz.min() < factor * CORE_BETA_THRESH)


# ---------------------------------------------------------------------------
# the two PS-Fdr selection rules
# ---------------------------------------------------------------------------

def clusso_support_cv(X, y, lambda_grid=None, n_folds=5, rng=None,
                      n_starts=DEFAULT_N_STARTS, tau=0.0):
    """
    Real-arm rule: CV-tuned CLUSSO, keep whatever it kept. Returns a (q,) bool.
    """
    fit = clusso_fit_cv(X, y, lambda_grid=lambda_grid, n_folds=n_folds,
                        rng=rng, n_starts=n_starts)
    return support_mask(fit['bet'], tau=tau)


def clusso_support_at_least_k(X, y, k, lambda_grid=None,
                              n_starts=DEFAULT_N_STARTS, tau=0.0):
    """
    Permuted-arm rule: a support of size >= k.

    PS-Fdr's step 2 fixes the number selected so the permuted arm cannot quietly
    collapse to nothing and drag the estimated false discovery count down with
    it. The published rule hits k exactly by walking a lasso path
    (ps_fdr.lasso_support_fixed_k). CLUSSO has no path -- every lambda is its own
    alternating solve -- and the map from lambda to support size is not reliably
    monotone, so an exact target may be unreachable and is certainly not
    bisectable.

    A floor is reachable, and it binds in the direction that matters: the failure
    the fixed count exists to prevent is the permuted arm selecting too FEW.
    Overshoot only makes the null stronger, which is the safe direction.

    Scans lambda from sparse to dense and stops at the first support reaching k,
    so the returned support is the sparsest one that clears the bar and the
    overshoot from any non-monotonicity is bounded.

    Failure mode, and it is not benign: if the grid runs out before reaching k,
    the returned support is smaller than k. That makes the permuted arm's
    selection frequencies too low, e0 too small, and Fdr-hat too small -- an
    ANTI-conservative error, silently under-controlling FDR. ``info['exhausted']``
    flags it, and any run must report the rate at which it fires.

    Returns ``(mask, info)`` where info carries ``lam``, ``k_realised``,
    ``exhausted`` and ``n_fits``.
    """
    X = np.asarray(X, dtype=float)
    q = X.shape[1]
    grid = DEFAULT_LAMBDA_GRID if lambda_grid is None else np.asarray(lambda_grid,
                                                                      dtype=float)
    k = int(k)

    if k <= 0:
        return (np.zeros(q, dtype=bool),
                {'lam': None, 'k_realised': 0, 'exhausted': False, 'n_fits': 0})

    best_mask = np.zeros(q, dtype=bool)
    best_lam = None
    n_fits = 0

    for lam in np.sort(grid)[::-1]:          # sparse -> dense
        fit = clusso_fit_multistart(X, y, lam, n_starts=n_starts)
        n_fits += 1
        mask = support_mask(fit['bet'], tau=tau)

        if mask.sum() > best_mask.sum():
            best_mask, best_lam = mask, float(lam)

        if mask.sum() >= k:
            return mask, {'lam': float(lam), 'k_realised': int(mask.sum()),
                          'exhausted': False, 'n_fits': n_fits}

    return best_mask, {'lam': best_lam, 'k_realised': int(best_mask.sum()),
                       'exhausted': True, 'n_fits': n_fits}
