# Controls and metrics from the two papers, so FDR-CLUSSO is scored on the same
# terms as the methods it is compared against.
#
# CONTROLS (the CLUSSO paper's comparison set, CLUSSO_Paper.pdf section 4.2):
#
#   naive    Lasso on each subject's tubule AVERAGES, 5-fold CV. Copied line for
#            line from the naive block inside core's CLUSSO_performance
#            (CLUSSO_Functions_Project1_6_16_23.py:217-266). That block is
#            inline there, so it can't be called alone without editing
#            src/core/. It is also what He et al. call CV-lasso.
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
#
# Bias and MSE need a coefficient estimate. A PS-Fdr arm returns a selected set
# and nothing else, so those two are None for the PS-Fdr arms rather than
# invented by a refit neither paper does.

import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LassoCV

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'core'))

from CLUSSO_Functions_Project1_6_16_23 import simulation_results  # noqa: E402

from clusso_select import clusso_fit_cv, support_mask              # noqa: E402


def naive_fit(X_avg, Y):
    """
    Core's naive averaging control, verbatim in logic. Returns the
    L1-normalised, 0.001-cut beta (what TPR/FPR/bias read) and the in-sample
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
    zero_inds = np.abs(beta) < 0.001
    beta[zero_inds] = 0.0

    beta_orig[zero_inds] = 0.0
    sse = float(np.sum((Y - X_avg @ beta_orig - intercept) ** 2))
    return {'bet': beta, 'mse': sse / len(Y)}


def oracle_fit(X_full, Y):
    """Full Information Structured Lasso, fitted the way stage 4 fits CLUSSO."""
    return clusso_fit_cv(X_full, Y, n_folds=5, rng=None, n_starts=5)


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


def structured_metrics(fit, Y, X, beta_star, truth):
    """
    Every metric for a structured-lasso fit (CLUSSO or the oracle). Bias and MSE
    come from core's own simulation_results, so they mean exactly what the CLUSSO
    paper's tables mean.
    """
    out = selection_metrics(support_mask(fit['bet']), truth)
    _, _, bias, mse = simulation_results(fit['bet'], beta_star, fit['alpha'], Y, X)
    out['bias'], out['mse'] = float(bias), float(mse)
    return out


def naive_metrics(X_avg, Y, beta_star, truth):
    fit = naive_fit(X_avg, Y)
    out = selection_metrics(fit['bet'] != 0, truth)
    bs = np.asarray(beta_star, dtype=float)
    out['bias'] = float(np.sum(np.abs(bs / np.sum(np.abs(bs)) - fit['bet'])))
    out['mse'] = float(fit['mse'])
    return out


def set_metrics(selected, truth):
    """A PS-Fdr arm: a selected set, so no bias or MSE."""
    mask = np.zeros(len(truth), dtype=bool)
    mask[np.asarray(selected, dtype=int)] = True
    out = selection_metrics(mask, truth)
    out['bias'], out['mse'] = None, None
    return out


# Report order and labels. The keys are what per_cohort records carry.
METRICS = [('fd', 'FD'), ('fn', 'FN'), ('fdp', 'FDP'), ('power', 'power'),
           ('fd_fn', 'FD+FN'), ('tpr', 'TPR%'), ('fpr', 'FPR%'),
           ('bias', 'bias'), ('mse', 'MSE'), ('n_sel', '|S|')]


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
