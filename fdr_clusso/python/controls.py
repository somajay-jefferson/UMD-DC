# Controls and metrics from the two papers, so FDR-CLUSSO is scored on the same
# terms as the methods it is compared against.
#
# CONTROLS (the CLUSSO paper's comparison set, CLUSSO_Paper.pdf section 4.2):
#
#   naive    Lasso on each subject's tubule AVERAGES, 5-fold CV. Copied line for
#            line from the naive block inside core's CLUSSO_performance
#            (CLUSSO_Functions_Project1_6_16_23.py:217-266). That block is
#            inline there, so it can't be called alone without editing
#            clusso/python/. It is also what He et al. call CV-lasso.
#
#   oracle   Full Information Structured Lasso: CLUSSO's own estimator on the
#            TRUE latent tubules, so no clustering error at all. Fitted exactly
#            as the stage 4 baseline fits CLUSSO -- clusso_fit_cv, 5 folds,
#            5 deterministic starts -- so the two differ only in the design.
#            Core draws a random alpha_init here instead; stage 1 is why we
#            don't.
#
# METRICS, every method:
#
#   He et al. (tables 1-2):   FD, FN, FDP, power, FD+FN
#   CLUSSO paper (sec. 4.2):  TPR %, FPR %, L1 bias of beta, in-sample MSE
#   Rubin's R code (both):    seconds per method, and the lambda it chose
#
# Bias and MSE need a coefficient estimate, and a PS-Fdr arm returns only a
# selected set. Random CLUSSO has the same problem and Rubin's code solves it by
# refitting STRUCTURED OLS on just the selected features
# (Random_CLUSSO_Functions_Simulate_Real_Data.R:283-311). `refit_metrics` does
# the same. Rubin takes Random CLUSSO's bias from its bootstrap-averaged betas
# instead; PS-Fdr keeps no betas, so its bias comes from the refit too.
#
# Clustering accuracy stays core's pooled-over-tubules number, which is also what
# Rubin's code computes. The paper's Appendix B averages per subject first. Doing
# that needs per-tubule labels, which only exist inside core's generator.

import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LassoCV

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'clusso', 'python'))

from CLUSSO_Functions_Project1_6_16_23 import simulation_results  # noqa: E402
from Mainfunction_albet_Structured_OLS import Mainfunction_albet_OLS  # noqa: E402

from clusso_select import (BETA_THRESH, DEFAULT_N_STARTS,         # noqa: E402
                           alpha_starts,
                           clusso_fit_cv, support_mask)


def naive_fit(X_avg, Y, thresh=BETA_THRESH):
    """
    Core's naive averaging control, verbatim in logic. Returns the
    L1-normalised, `thresh`-cut beta (what TPR/FPR/bias read) and the in-sample
    MSE on the original scale (what core reports).
    """
    n_subj = len(Y)
    p_feat = X_avg.shape[1]

    X_avg_mu = X_avg.mean(axis=0)
    X_avg_c = X_avg - X_avg_mu
    X_avg_scale = np.sqrt((X_avg_c ** 2).mean(axis=0))
    X_avg_scale = np.where(X_avg_scale > 1e-10, X_avg_scale, 1.0)
    X_avg_std = X_avg_c / X_avg_scale

    Y_mu = Y.mean()
    Y_c = Y - Y_mu

    lambda_max = float(np.max(np.abs(X_avg_std.T @ Y_c)) / n_subj)
    eps = 0.01 if n_subj >= p_feat else 0.0001
    lambda_min = max(lambda_max * eps, 1e-10)
    lambdas_R = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), 100))

    avg_model = LassoCV(cv=5, alphas=lambdas_R / 2.0,
                        fit_intercept=False, max_iter=100_000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        avg_model.fit(X_avg_std, Y_c)

    beta_orig = avg_model.coef_ / X_avg_scale
    intercept = float(Y_mu - X_avg_mu @ beta_orig)

    l1 = np.sum(np.abs(beta_orig))
    beta = beta_orig / l1 if l1 > 0 else beta_orig.copy()
    # core's naive block cuts at 0.001; `thresh` lets every method share one cut,
    # as the Random CLUSSO paper applies 0.01 to all four of its methods.
    zero_inds = np.abs(beta) < thresh
    beta[zero_inds] = 0.0

    beta_orig[zero_inds] = 0.0
    sse = float(np.sum((Y - X_avg @ beta_orig - intercept) ** 2))
    # lam on R's glmnet scale, like every other lambda this directory records
    return {'bet': beta, 'mse': sse / len(Y), 'lam': float(avg_model.alpha_) * 2.0,
            'beta_orig': beta_orig, 'intercept': intercept}


def oracle_fit(X_full, Y, thresh=BETA_THRESH, lambda_grid=None):
    """Full Information Structured Lasso, fitted the way stage 4 fits CLUSSO."""
    return clusso_fit_cv(X_full, Y, lambda_grid=lambda_grid, n_folds=5,
                         rng=None, n_starts=5, thresh=thresh)


# --- held-out validation, as Random CLUSSO scores prediction -----------------
#
# Random_CLUSSO_Data_Analysis.R (Supplemental_Code.zip) and paper section 4:
# split subjects 80/20, fit every method on the 80%, score MSE on the 20%.
# Structured methods predict with test-centered Y and test-centered X (no
# intercept); naive predicts raw test averages with its training intercept.
# Cluster labels are arbitrary, so a structured fit is scored with alpha in
# both orders and the smaller MSE kept -- the paper's "992 or 2670".
#
# One deliberate difference: Rubin refits Random CLUSSO's structured OLS on
# the TEST split and scores it there, which fits coefficients to the data they
# are scored on. The refit here is trained on the training split only.

def _test_scores(mse, mse_other, Y_test):
    var = float(np.var(Y_test))
    return {'test_mse': float(mse), 'test_rmse': float(np.sqrt(mse)),
            'test_r2': (1.0 - float(mse) / var) if var > 0 else None,
            'test_mse_other_order': (None if mse_other is None
                                     else float(mse_other))}


def structured_test(alpha, bet, X_test, Y_test, flip=True):
    """
    Held-out MSE for a structured fit, Rubin's way. flip=True also scores
    alpha reversed and keeps the smaller, for fits on CLUSTERED designs whose
    labels mean nothing across a split. The oracle's labels are the true ones,
    so it passes flip=False.
    """
    alpha = np.asarray(alpha, float)
    bet = np.asarray(bet, float)
    Yc = Y_test - Y_test.mean()
    Xc = X_test - X_test.mean(axis=2, keepdims=True)

    def mse(a):
        pred = np.einsum('p,pqi,q->i', a, Xc, bet)
        return float(np.mean((Yc - pred) ** 2))

    m = mse(alpha)
    if not flip or alpha.size != 2:
        return _test_scores(m, None, Y_test)
    m2 = mse(alpha[::-1])
    return _test_scores(min(m, m2), max(m, m2), Y_test)


def selection_metrics(mask, truth):
    """He et al.'s set, plus TPR/FPR in the CLUSSO paper's percent units."""
    mask = np.asarray(mask, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    fd = int((mask & ~truth).sum())
    fn = int((~mask & truth).sum())
    n_sel = int(mask.sum())
    n_true = int(truth.sum())
    tp = n_sel - fd
    return {
        'n_sel': n_sel, 'fd': fd, 'fn': fn, 'fd_fn': fd + fn,
        'fdp': fd / n_sel if n_sel else 0.0,     # 0 when empty, as in He et al.
        'power': tp / n_true if n_true else 0.0,
        'tpr': 100.0 * tp / max(n_true, 1),
        'fpr': 100.0 * fd / max(int((~truth).sum()), 1),
    }


def structured_metrics(fit, Y, X, beta_star, truth, test=None, flip=True):
    """
    Every metric for a structured-lasso fit (CLUSSO or the oracle). Bias and MSE
    come from core's own simulation_results, so they mean exactly what the CLUSSO
    paper's tables mean.
    """
    out = selection_metrics(support_mask(fit['bet']), truth)
    _, _, bias, mse = simulation_results(fit['bet'], beta_star, fit['alpha'], Y, X)
    out['bias'], out['mse'] = float(bias), float(mse)
    out['lam'] = float(fit['lam'])
    if test is not None:
        out.update(structured_test(fit['alpha'], fit['bet'], *test, flip=flip))
    return out


def naive_metrics(X_avg, Y, beta_star, truth, test=None, thresh=BETA_THRESH):
    fit = naive_fit(X_avg, Y, thresh=thresh)
    out = selection_metrics(fit['bet'] != 0, truth)
    bs = np.asarray(beta_star, dtype=float)
    out['bias'] = float(np.sum(np.abs(bs / np.sum(np.abs(bs)) - fit['bet'])))
    out['mse'] = float(fit['mse'])
    out['lam'] = fit['lam']
    if test is not None:
        X_te, Y_te = test
        m = float(np.mean((Y_te - X_te @ fit['beta_orig'] - fit['intercept']) ** 2))
        out.update(_test_scores(m, None, Y_te))
    return out


def refit_sols(X, Y, selected, n_starts=DEFAULT_N_STARTS):
    """
    Structured OLS on the selected columns only, as Random CLUSSO does. Rubin
    starts from rnorm draws; this uses the same deterministic starts CLUSSO
    does and keeps the lowest in-sample SSE. Returns a full-length (q,) beta,
    zero off the selected set, so it scores against the full design.
    """
    P, q, n = X.shape
    selected = np.asarray(selected, dtype=int)
    bet_full = np.zeros(q)
    if selected.size == 0:
        return {'alpha': np.zeros(P), 'bet': bet_full}
    Xs = X[:, selected, :]
    best, best_sse = None, np.inf
    for a0 in alpha_starts(P, n_starts):
        fit = Mainfunction_albet_OLS(Xs, Y, a0, np.ones(selected.size))
        sse = simulation_results(fit['bet'], np.ones(selected.size),
                                 fit['alpha'], Y, Xs)[3]
        if sse < best_sse:
            best, best_sse = fit, sse
    bet_full[selected] = best['bet']
    return {'alpha': best['alpha'], 'bet': bet_full}


def refit_metrics(selected, X, Y, beta_star, truth, test=None):
    """A PS-Fdr arm: selection metrics from the set, bias and MSE from the refit."""
    mask = np.zeros(len(truth), dtype=bool)
    mask[np.asarray(selected, dtype=int)] = True
    out = selection_metrics(mask, truth)
    fit = refit_sols(X, Y, selected)
    _, _, bias, mse = simulation_results(fit['bet'], beta_star, fit['alpha'], Y, X)
    out['bias'], out['mse'] = float(bias), float(mse)
    if test is not None:
        # The refit is trained on X, Y -- the training split -- never on test.
        out.update(structured_test(fit['alpha'], fit['bet'], *test))
    return out


# Report order and labels. The keys are what per_cohort records carry.
METRICS = [('fd', 'FD'), ('fn', 'FN'), ('fdp', 'FDP'), ('power', 'power'),
           ('fd_fn', 'FD+FN'), ('tpr', 'TPR%'), ('fpr', 'FPR%'),
           ('bias', 'bias'), ('mse', 'MSE'), ('n_sel', '|S|'),
           ('test_mse', 'test MSE'), ('test_rmse', 'test RMSE'),
           ('test_r2', 'test R2'), ('sec', 'sec')]


def summarize(records):
    """Mean and median of every metric over cohorts. None where undefined."""
    out = {}
    for key, _ in METRICS:
        vals = [r.get(key) for r in records]
        if any(v is None for v in vals):
            out[key] = {'mean': None, 'median': None}
            continue
        a = np.asarray(vals, dtype=float)
        out[key] = {'mean': float(a.mean()), 'median': float(np.median(a))}
    return out
