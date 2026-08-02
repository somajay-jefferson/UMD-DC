# Author: Jeremy Rubin
# Date: 6/16/23
# Functions to run CLUSSO as well as generate performance metrics

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.mixture import GaussianMixture

from Mainfunction_albet import Mainfunction_albet
from SLasso_MSE import slasso_mse, CV_make_folds, lambda_CV_mse


def calc_ab(sigma_sq_m, mu):
    """Compute (a, b) for discrete-uniform distribution given desired mean and variance."""
    H = int(np.sqrt(12 * sigma_sq_m + 1))
    if H % 2 == 0:
        H -= 1
    a = mu - (H - 1) / 2
    b = mu + (H - 1) / 2
    return a, b


def generate_design_matrices(n, mu, sigma_sq_m, sigma_sq,
                              x1, x2, sigma1, sigma2, sigma_r,
                              alpha_star, beta_star, w):
    """
    Generate synthetic scalar-on-matrix regression data.

    Parameters
    ----------
    n           : number of subjects
    mu          : mean number of objects per subject
    sigma_sq_m  : variance of number of objects per subject
    sigma_sq    : outcome noise variance
    x1, x2      : cluster means (atrophic / normal tubule)
    sigma1, sigma2 : cluster variances
    sigma_r     : measurement error variance
    alpha_star  : (P,) true cluster weights
    beta_star   : (q,) true feature coefficients
    w           : (n,) per-subject atrophic tubule weights

    Returns
    -------
    Y                   : (n,) outcomes
    X_avg               : (n, q) naive-average design matrix
    clust_X             : (P, q, n) cluster-averaged design matrices
    X_full              : (P, q, n) full-information design matrices
    clustering_accuracy : float
    no_clust            : bool - True if any subject had an empty cluster
    """
    H = int(np.sqrt(12 * sigma_sq_m + 1))
    if H % 2 == 0:
        H -= 1
    a = int(mu - (H - 1) / 2)
    b = int(mu + (H - 1) / 2)

    p_vec      = np.random.randint(a, b + 1, size=n)
    P          = len(alpha_star)
    q          = len(beta_star)
    alpha_star = np.asarray(alpha_star, dtype=float).flatten()
    beta_star  = np.asarray(beta_star,  dtype=float).flatten()

    X_avg  = np.zeros((n, q))
    X_full = np.ones((P, q, n))
    Y      = np.zeros(n)

    cluster_labels = []
    raw_X          = []

    for j in range(n):
        Xi_star = np.zeros((P, q))
        Xi_star[0, :] = np.random.normal(x1, np.sqrt(sigma1), size=q)
        Xi_star[1, :] = np.random.normal(x2, np.sqrt(sigma2), size=q)

        probs       = np.array([w[j], 1.0 - w[j]])
        resamp_rows = np.random.choice(P, size=p_vec[j], replace=True, p=probs)
        resamp_Xi   = (Xi_star[resamp_rows, :]
                       + np.random.normal(0, np.sqrt(sigma_r), size=(p_vec[j], q)))

        cluster_labels.append(resamp_rows + 1)   # 1-indexed (matches R)
        raw_X.append(resamp_Xi)

        Xi_star_w        = Xi_star.copy()
        Xi_star_w[0, :] *= w[j]
        Xi_star_w[1, :] *= (1.0 - w[j])

        Y[j]            = (alpha_star @ Xi_star_w @ beta_star
                           + np.random.normal(0, np.sqrt(sigma_sq)))
        X_avg[j, :]     = resamp_Xi.mean(axis=0)
        X_full[:, :, j] = Xi_star_w

    # ── Joint GMM clustering across all subjects ──────────────────────────────
    X_all       = np.vstack(raw_X)
    cluster_all = np.concatenate(cluster_labels)

    # GaussianMixture with full covariance is the closest Python equivalent to
    # R's Mclust(X, G=P).  n_init=10 gives multiple random starts like Mclust;
    # tol and max_iter match Mclust's EM defaults.
    gmm = GaussianMixture(n_components=P, covariance_type='full',
                          n_init=10, max_iter=1000, tol=1e-5)
    gmm.fit(X_all)
    NMM_out = gmm.predict(X_all) + 1   # 1-indexed

    cluster_acc_check = cluster_all.copy()
    NMM_acc_check     = NMM_out.copy()

    # Attach true + estimated labels to each subject's matrix
    NMM_remaining   = NMM_out.copy()
    clust_remaining = cluster_all.copy()
    for i in range(n):
        pi         = p_vec[i]
        raw_X[i]   = np.column_stack([clust_remaining[:pi],
                                      NMM_remaining[:pi],
                                      raw_X[i]])
        NMM_remaining   = NMM_remaining[pi:]
        clust_remaining = clust_remaining[pi:]

    # Clustering accuracy (best of two label correspondences)
    c1 = np.mean(((cluster_acc_check == 1) & (NMM_acc_check == 1)) |
                 ((cluster_acc_check == 2) & (NMM_acc_check == 2)))
    c2 = np.mean(((cluster_acc_check == 1) & (NMM_acc_check == 2)) |
                 ((cluster_acc_check == 2) & (NMM_acc_check == 1)))
    clustering_accuracy = max(c1, c2)

    # ── Build cluster-averaged matrices ───────────────────────────────────────
    clust_X  = np.ones((P, q, n))
    no_clust = False

    for i in range(n):
        clust1 = raw_X[i][raw_X[i][:, 1] == 1]
        clust2 = raw_X[i][raw_X[i][:, 1] == 2]

        # R's is.null(dim(x)) is TRUE when a matrix has exactly 1 row and R
        # coerces it to a vector.  Match that: flag 0 or 1 rows per cluster.
        if clust1.shape[0] <= 1 or clust2.shape[0] <= 1:
            no_clust = True
            clust_X[:, :, i] = 0.0
        else:
            w1 = clust1.shape[0] / raw_X[i].shape[0]
            clust_X[0, :, i] = w1          * clust1[:, 2:].mean(axis=0)
            clust_X[1, :, i] = (1.0 - w1) * clust2[:, 2:].mean(axis=0)

    return Y, X_avg, clust_X, X_full, clustering_accuracy, no_clust


def simulation_results(beta_hat, beta_star, alpha_hat, Y, X):
    """
    Compute TPR, FPR, L1 bias, and MSE for structured lasso estimates.

    X : (p, q, n) array
    Returns numpy array [TPR, FPR, bias, MSE].
    """
    beta_hat  = np.asarray(beta_hat).flatten()
    beta_star = np.asarray(beta_star).flatten()
    alpha_hat = np.asarray(alpha_hat).flatten()

    beta_star = beta_star / np.sum(np.abs(beta_star))

    nz_hat  = np.where(beta_hat  != 0)[0]
    nz_star = np.where(beta_star != 0)[0]
    z_star  = np.where(beta_star == 0)[0]

    TPR  = (len(np.intersect1d(nz_hat, nz_star)) / max(len(nz_star), 1)) * 100
    FPR  = (len(np.intersect1d(nz_hat, z_star))  / max(len(z_star),  1)) * 100
    bias = float(np.sum(np.abs(beta_star - beta_hat)))

    Y_c    = Y - Y.mean()
    X_mean = X.mean(axis=2)   # (p, q)
    n      = X.shape[2]
    SSE    = 0.0
    for i in range(n):
        Xi  = X[:, :, i] - X_mean
        SSE += (Y_c[i] - alpha_hat @ Xi @ beta_hat) ** 2

    return np.array([TPR, FPR, bias, SSE / n])


def CLUSSO_performance(n, mu, sigma_sq_m, sigma_sq,
                        x1, x2, sigma1, sigma2, sigma_r,
                        alpha_star, beta_star, lambda_grid, w):
    """
    Compare CLUSSO vs naive averaging vs full-information structured lasso.

    Returns
    -------
    performance         : pd.DataFrame (4 x 3) with rows TPR/FPR/bias/MSE,
                          columns CLUSSO / Average balancing / Full information
    lambdas             : [lambda_clust, lambda_full]
    clustering_accuracy : float
    All three are None if the simulation should be discarded (empty cluster).
    """
    q         = len(beta_star)
    beta_init = np.random.normal(size=q)

    Y, X_avg, clust_X, X_full, clustering_accuracy, no_clust = generate_design_matrices(
        n, mu, sigma_sq_m, sigma_sq, x1, x2, sigma1, sigma2, sigma_r,
        alpha_star, beta_star, w
    )

    if not no_clust:
        for i in range(n):
            if np.isnan(clust_X[:, :, i]).any():
                no_clust = True
                break

    if no_clust:
        return None, None, None

    # ── Naive averaging: equivalent to R's cv.glmnet(X.avg, Y, nfolds=5) ─────
    #
    # R's cv.glmnet standardizes X (intercept=TRUE, standardize=TRUE) and uses
    # a data-driven 100-point log-spaced lambda path.  Objective: (1/n)*RSS.
    # sklearn LassoCV uses (1/2n)*RSS, so alpha_sklearn = lambda_R / 2.

    n_subj  = len(Y)
    p_feat  = X_avg.shape[1]

    # Standardize X_avg (population std, matching glmnet standardize=TRUE)
    X_avg_mu    = X_avg.mean(axis=0)
    X_avg_c     = X_avg - X_avg_mu
    X_avg_scale = np.sqrt((X_avg_c ** 2).mean(axis=0))
    X_avg_scale = np.where(X_avg_scale > 1e-10, X_avg_scale, 1.0)
    X_avg_std   = X_avg_c / X_avg_scale

    # Center Y (glmnet intercept=TRUE centers the response)
    Y_mu = Y.mean()
    Y_c  = Y - Y_mu

    # Compute glmnet-style lambda path
    lambda_max = float(np.max(np.abs(X_avg_std.T @ Y_c)) / n_subj)
    eps        = 0.01 if n_subj >= p_feat else 0.0001
    lambda_min = max(lambda_max * eps, 1e-10)
    lambdas_R  = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), 100))

    # LassoCV with corrected alphas (lambda_R / 2) and fit_intercept=False
    # because we already centered X and Y manually
    avg_model = LassoCV(cv=5, alphas=lambdas_R / 2.0,
                        fit_intercept=False, max_iter=100_000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        avg_model.fit(X_avg_std, Y_c)

    # Back-transform coefficients to original scale
    beta_hat_avg_orig = avg_model.coef_ / X_avg_scale
    # Recover intercept: beta0 = mean(Y) - mean(X)' @ beta
    intercept         = float(Y_mu - X_avg_mu @ beta_hat_avg_orig)

    beta_hat_avg  = beta_hat_avg_orig / np.sum(np.abs(beta_hat_avg_orig))
    zero_inds     = np.abs(beta_hat_avg) < 0.001
    beta_hat_avg[zero_inds] = 0.0

    beta_star_norm = np.asarray(beta_star, dtype=float) / np.sum(np.abs(beta_star))
    nz_hat  = np.where(beta_hat_avg    != 0)[0]
    nz_star = np.where(beta_star_norm  != 0)[0]
    z_star  = np.where(beta_star_norm  == 0)[0]

    TPR_avg  = (len(np.intersect1d(nz_hat, nz_star)) / max(len(nz_star), 1)) * 100
    FPR_avg  = (len(np.intersect1d(nz_hat, z_star))  / max(len(z_star),  1)) * 100
    bias_avg = float(np.sum(np.abs(beta_star_norm - beta_hat_avg)))

    beta_hat_avg_orig[zero_inds] = 0.0
    SSE = sum((Y[r] - beta_hat_avg_orig @ X_avg[r, :] - intercept) ** 2
              for r in range(len(Y)))
    MSE_avg = SSE / len(Y)

    # ── Full information structured lasso ─────────────────────────────────────
    P          = X_full.shape[0]
    alpha_init = np.random.normal(size=P)

    mse_full    = [lambda_CV_mse(X_full, Y, alpha_init, beta_init, l) for l in lambda_grid]
    lambda_full = lambda_grid[int(np.argmin(mse_full))]

    res_full   = Mainfunction_albet(X_full, Y, alpha_init, beta_init, lambda_full)
    stats_full = simulation_results(res_full['bet'], beta_star,
                                    res_full['alpha'], Y, X_full)
    TPR_full, FPR_full, bias_full, MSE_full = stats_full

    # ── CLUSSO ────────────────────────────────────────────────────────────────
    alpha_init = np.random.normal(size=clust_X.shape[0])

    mse_clust    = [lambda_CV_mse(clust_X, Y, alpha_init, beta_init, l) for l in lambda_grid]
    lambda_clust = lambda_grid[int(np.argmin(mse_clust))]

    res_clust   = Mainfunction_albet(clust_X, Y, alpha_init, beta_init, lambda_clust)
    stats_clust = simulation_results(res_clust['bet'], beta_star,
                                     res_clust['alpha'], Y, clust_X)
    TPR_clust, FPR_clust, bias_clust, MSE_clust = stats_clust

    performance = pd.DataFrame(
        [[TPR_clust,  TPR_avg,  TPR_full],
         [FPR_clust,  FPR_avg,  FPR_full],
         [bias_clust, bias_avg, bias_full],
         [MSE_clust,  MSE_avg,  MSE_full]],
        index   = ['TPR', 'FPR', 'L1 norm of bias', 'MSE'],
        columns = ['CLUSSO', 'Average balancing', 'Full information']
    )

    return performance, [lambda_clust, lambda_full], clustering_accuracy
