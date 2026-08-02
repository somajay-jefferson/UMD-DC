# Date: 7/20/23
# Coefficient function for computing structured lasso fit

import warnings
import numpy as np
from sklearn.linear_model import Lasso


def coefficient(x, y, Lamd=None):
    """
    Lasso coefficient estimation equivalent to R's glmnet(x, y, alpha=1).

    Matches R's glmnet defaults:
      intercept=TRUE  : centers y and each column of x before fitting
      standardize=TRUE: scales x columns by population std sqrt(mean((x-mean)^2))

    Objective correction
    --------------------
    R glmnet  minimises  (1/n)   * ||y - xb||^2  +  lambda * ||b||_1
    sklearn   minimises  (1/2n)  * ||y - xb||^2  +  alpha  * ||b||_1
    Match via:  alpha_sklearn = lambda_R / 2

    Parameters
    ----------
    Lamd : float or None
        If given, fits at that lambda value (R glmnet scale).
        If None, computes a glmnet-style 100-point path and selects by BIC,
        equivalent to R's:  which.min(deviance(fit) + fit$df * log(n))

    Returns
    -------
    dict with:
        'b'    : coefficient vector (q, 1) on the original scale
        'lamd' : lambda path in R glmnet scale (only when Lamd is None)
    """
    n, p = x.shape

    # Center y  (glmnet intercept=TRUE)
    y_mean = y.mean()
    y_c    = y - y_mean

    # Center and scale x  (glmnet standardize=TRUE, population std)
    x_mu    = x.mean(axis=0)
    x_c     = x - x_mu
    x_scale = np.sqrt((x_c ** 2).mean(axis=0))          # sqrt(sum((x-mu)^2) / n)
    x_scale = np.where(x_scale > 1e-10, x_scale, 1.0)  # guard zero-variance columns
    x_std   = x_c / x_scale

    if Lamd is not None:
        # alpha_sklearn = lambda_R / 2
        sk_alpha = max(float(Lamd) / 2.0, 1e-10)
        model = Lasso(alpha=sk_alpha, fit_intercept=False, max_iter=100_000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(x_std, y_c)
        b = (model.coef_ / x_scale).reshape(-1, 1)
        return {'b': b}

    # ── Fit glmnet-style path, select by BIC ─────────────────────────────────
    # lambda_max: smallest lambda that makes all coefficients zero
    # From KKT conditions: lambda_max = max|X_std' y_c| / n
    lambda_max = float(np.max(np.abs(x_std.T @ y_c)) / n)

    # glmnet default: eps = 0.01 when n >= p, else 0.0001
    eps        = 0.01 if n >= p else 0.0001
    lambda_min = max(lambda_max * eps, 1e-10)

    # 100 log-spaced lambda values (glmnet default nlambda = 100)
    lambdas_R = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), 100))

    coef_path = []
    for lam_R in lambdas_R:
        m = Lasso(alpha=lam_R / 2.0, fit_intercept=False, max_iter=100_000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(x_std, y_c)
        coef_path.append(m.coef_ / x_scale)    # back-transform to original scale
    coef_path = np.array(coef_path).T           # (q, 100)

    # BIC = deviance + df * log(n)
    # For Gaussian, deviance = RSS on centered y (matches R's deviance(fit))
    residuals = y_c[:, None] - x_c @ coef_path  # (n, 100)  centered residuals
    rss       = (residuals ** 2).sum(axis=0)
    df        = (coef_path != 0).sum(axis=0)
    bic       = rss + df * np.log(n)
    best_idx  = int(np.argmin(bic))

    b = coef_path[:, best_idx].reshape(-1, 1)
    return {'b': b, 'lamd': lambdas_R}
