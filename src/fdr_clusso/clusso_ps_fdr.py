# CLUSSO plugged into PS-Fdr's estimator seam.
#
# `ps_fdr` calls its estimator in exactly four places and never inspects it
# otherwise (see the seam comment in src/fdr/ps_fdr.py). This module supplies
# those four methods for CLUSSO, so PS-Fdr runs unchanged with CLUSSO where the
# lasso was.
#
# Two things this file is responsible for, and nothing else is:
#
#   Axis order. `clusso_select` speaks CLUSSO's native (P, q, n) -- clusters,
#   features, subjects. `ps_fdr` resamples along axis 0 and counts features with
#   X.shape[-1], so it needs (n, P, q). The transpose happens here, at the
#   boundary, in both directions. check.py c12 pins it.
#
#   Counters. `clusso_support_exact_k` reports whether the lambda grid ran out
#   before reaching k, which is an anti-conservative failure and silent unless
#   someone tallies it. The seam's signature returns a plain mask, so the tally
#   lives on the selector object and the caller reads it afterwards.
#
# Kept separate from clusso_select.py on purpose: that module deliberately
# imports only from src/core/, and this one needs src/fdr/.

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'fdr'))

from ps_fdr import ps_fdr                                   # noqa: E402

from clusso_select import (DEFAULT_LAMBDA_GRID, DEFAULT_N_STARTS,  # noqa: E402
                           NULL_LAMBDA_GRID, clusso_cv_lambda,
                           clusso_support_at_lambda,
                           clusso_support_at_least_k, clusso_support_cv,
                           clusso_support_exact_k)


def to_ps_fdr(clust_X):
    """CLUSSO's native (P, q, n) -> the (n, P, q) ps_fdr resamples along axis 0."""
    return np.ascontiguousarray(np.asarray(clust_X, dtype=float).transpose(2, 0, 1))


def to_clusso(X):
    """The inverse: (n, P, q) -> (P, q, n)."""
    return np.ascontiguousarray(np.asarray(X, dtype=float).transpose(1, 2, 0))


class ClussoSelector:
    """
    CLUSSO behind PS-Fdr's four-method seam.

    Every method takes X in ps_fdr's (n, P, q) orientation and transposes back
    before calling into `clusso_select`, which is the only module that knows
    CLUSSO's real shape.

    Parameters mirror `clusso_select`'s defaults. `n_starts` is the one that
    matters: it is what makes the selection rule deterministic, which stability
    selection requires and stage1_noise.py measured.
    """

    def __init__(self, lambda_grid=None, n_starts=DEFAULT_N_STARTS, tau=0.0,
                 n_folds=5, null_lambda_grid=None):
        self.lambda_grid = (DEFAULT_LAMBDA_GRID if lambda_grid is None
                            else np.asarray(lambda_grid, dtype=float))
        # The permuted arm walks a COARSER ladder than the real arm. Safe only
        # because truncation cuts back to exactly k however far the scan
        # overshot, and immaterial because the null arm's output is a curve over
        # sorted RANKS rather than a list of features -- see NULL_LAMBDA_GRID in
        # clusso_select.py. The real arm keeps the fine grid, where identity
        # does matter.
        self.null_lambda_grid = (NULL_LAMBDA_GRID if null_lambda_grid is None
                                 else np.asarray(null_lambda_grid, dtype=float))
        self.n_starts = n_starts
        self.tau = tau
        self.n_folds = n_folds

        # Tallies over every permuted-arm fit this selector performs.
        self.n_fixed_k_calls = 0
        self.n_exhausted = 0
        self.n_truncated = 0
        self.total_fits = 0

    # -- the seam ----------------------------------------------------------
    #
    # `n_folds` and `random_state` arrive from ps_fdr on two of these. CLUSSO's
    # CV takes its own fold count and is pure by construction (rng=None gives
    # contiguous folds), so random_state is accepted and ignored rather than
    # threaded anywhere -- there is no stochastic component left for it to seed.

    def cv_lambda(self, X, y, n_folds, random_state):
        return clusso_cv_lambda(to_clusso(X), y, lambda_grid=self.lambda_grid,
                                n_folds=self.n_folds, rng=None,
                                n_starts=self.n_starts)

    def at_lambda(self, X, y, lam):
        self.total_fits += 1
        return clusso_support_at_lambda(to_clusso(X), y, lam,
                                        n_starts=self.n_starts, tau=self.tau)

    def support_cv(self, X, y, n_folds, random_state):
        return clusso_support_cv(to_clusso(X), y, lambda_grid=self.lambda_grid,
                                 n_folds=self.n_folds, rng=None,
                                 n_starts=self.n_starts, tau=self.tau)

    def support_fixed_k(self, X, y, k, rule):
        if rule == 'exact':
            mask, info = clusso_support_exact_k(
                to_clusso(X), y, k, lambda_grid=self.null_lambda_grid,
                n_starts=self.n_starts, tau=self.tau)
            if info['k_before_truncation'] > int(k):
                self.n_truncated += 1
        elif rule == 'atleast':
            mask, info = clusso_support_at_least_k(
                to_clusso(X), y, k, lambda_grid=self.null_lambda_grid,
                n_starts=self.n_starts, tau=self.tau)
        else:
            raise ValueError(
                f"null_rule={rule!r} has no CLUSSO counterpart; "
                "'nearest' needs a coefficient path. Use 'exact' or 'atleast'.")

        self.n_fixed_k_calls += 1
        self.n_exhausted += bool(info['exhausted'])
        self.total_fits += info['n_fits']
        return mask

    # -- diagnostics -------------------------------------------------------

    @property
    def exhausted_rate(self):
        """
        Fraction of permuted-arm fits that ran out of lambda grid before
        reaching k.

        This is the one failure truncation does NOT fix. A short support makes
        e0 too small and Fdr-hat too small -- anti-conservative, and it does not
        announce itself. Any run must report this.
        """
        if self.n_fixed_k_calls == 0:
            return 0.0
        return self.n_exhausted / self.n_fixed_k_calls

    @property
    def mean_fits_per_target(self):
        """Mean CLUSSO fits spent per permuted-arm selection. Cost, and a hint
        about whether the lambda grid wants coarsening."""
        if self.n_fixed_k_calls == 0:
            return 0.0
        return self.total_fits / self.n_fixed_k_calls


def clusso_ps_fdr(clust_X, y, selector=None, **kw):
    """
    PS-Fdr with CLUSSO as the selection rule, on a frozen design tensor.

    `clust_X` is CLUSSO's native (P, q, n). It is built ONCE, by whoever calls
    this, and reused for every resample -- the clustering does not happen again
    inside the loop. That is what makes the composition possible at all, and it
    means any false discovery guarantee obtained here is conditional on that one
    clustering solution. See REVIEW.md section 4.

    Extra keywords go straight to `ps_fdr`. The two that decide the experiment:

      null_mode='fixed', null_rule='exact'   the published rule, ported: scan to
                                             a support of at least k, then keep
                                             the k largest coefficients.
      null_mode='fixed_lambda'               one CV-tuned lambda everywhere,
                                             nothing targeting a count.

    Returns ``(res, selector)`` -- ps_fdr's dict, and the selector so the caller
    can read `exhausted_rate` off it.
    """
    selector = ClussoSelector() if selector is None else selector
    res = ps_fdr(to_ps_fdr(clust_X), y, selector=selector, **kw)
    return res, selector
