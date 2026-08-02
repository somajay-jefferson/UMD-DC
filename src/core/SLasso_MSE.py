# Author: Jeremy Rubin
# Date: 7/20/23 (original R), 2026 (Python port)
# Functions to compute MSE for structured lasso across a grid of regularization
# parameters with cross-validation.
#
# Port of SLasso_MSE.R (shipped in clussocode.zip). The R original is the
# reference for every behavioural choice below; deviations are called out.

import numpy as np

from Mainfunction_albet import Mainfunction_albet


def slasso_mse(X_train, Y_train, X_test, Y_test, alpha_init, beta_init, lam):
    """
    Fit the structured lasso on the training subjects, then score it on the
    held-out subjects.

    Equivalent to R's `slasso.mse`. Two details that are easy to get wrong and
    that the R original does deliberately:

      * `Y.test` is **mean-centred only** (`scale(Y.test, scale = FALSE)`),
        never divided by its standard deviation.
      * each test subject's matrix is centred by the mean taken over the
        **test fold's own subjects** (`apply(X.test, c(1,2), mean)`), not by a
        mean carried over from the training fold.

    Parameters
    ----------
    X_train : (p, q, n_train) array
    Y_train : (n_train,) array
    X_test  : (p, q, n_test) array
    Y_test  : (n_test,) array
    alpha_init, beta_init : non-zero initial vectors, shapes (p,) and (q,)
    lam     : regularisation parameter, on R's glmnet scale

    Returns
    -------
    float : mean squared error over the test subjects
    """
    fit = Mainfunction_albet(X_train, Y_train, alpha_init, beta_init, lam)
    alpha_hat = fit['alpha']
    beta_hat = fit['bet']

    Y_test = np.asarray(Y_test, dtype=float).flatten()
    n_test = X_test.shape[2]

    # Mean-centre the test outcomes (scale = FALSE in R: centre, do not scale)
    Y_test_c = Y_test - Y_test.mean()

    # Mean of the test-fold matrices, taken over the subject axis -> (p, q)
    X_mean_test = X_test.mean(axis=2)

    SSE = 0.0
    for i in range(n_test):
        # R mutates X.test in place here; we do not touch the caller's array
        Xi = X_test[:, :, i] - X_mean_test
        SSE += (Y_test_c[i] - alpha_hat @ Xi @ beta_hat) ** 2

    return float(SSE / n_test)


def CV_make_folds(n_train):
    """
    Partition subject indices into five cross-validation folds.

    Equivalent to R's `CV.make.folds`. Folds one through four each hold
    ``floor(n_train / 5)`` subjects drawn without replacement; fold five takes
    whatever is left, so it is the largest whenever ``n_train`` is not a
    multiple of five.

    Uses the legacy ``np.random`` global stream so that a caller's
    ``np.random.seed(...)`` governs fold assignment, matching how the rest of
    this codebase seeds itself.

    Deviation from R: ``n_train < 5`` raises instead of silently producing
    empty test folds, which in R yields a divide-by-zero and a NaN MSE.

    Returns
    -------
    list of five sorted integer arrays of 0-based subject indices
    """
    n_train = int(n_train)
    if n_train < 5:
        raise ValueError(
            f"need at least 5 subjects for 5-fold CV, got {n_train}"
        )

    size = n_train // 5
    remaining = np.arange(n_train)
    folds = []

    for _ in range(4):
        picked = np.random.choice(remaining, size=size, replace=False)
        folds.append(np.sort(picked))
        remaining = np.setdiff1d(remaining, picked)

    # Fold five is the remainder, exactly as R's final setdiff
    folds.append(np.sort(remaining))
    return folds


def lambda_CV_mse(X_train, Y_train, alpha_init, beta_init, lam):
    """
    Five-fold cross-validated MSE for one regularisation parameter.

    Equivalent to R's `lambda.CV.mse`: build the folds once, then for each fold
    train on the other four and score on the held-out one, and average the five
    results. The same ``alpha_init`` / ``beta_init`` are reused for every fold,
    as in the R original -- they are not redrawn per fold.

    Typical use is one call per candidate lambda:

        mse = [lambda_CV_mse(X, y, a0, b0, l) for l in lambda_grid]
        best = lambda_grid[int(np.argmin(mse))]

    Parameters
    ----------
    X_train : (p, q, n) array
    Y_train : (n,) array
    alpha_init, beta_init : non-zero initial vectors, shapes (p,) and (q,)
    lam     : regularisation parameter, on R's glmnet scale

    Returns
    -------
    float : mean of the five fold MSEs
    """
    Y_train = np.asarray(Y_train, dtype=float).flatten()
    folds = CV_make_folds(X_train.shape[2])
    n_folds = len(folds)

    mse_vec = np.zeros(n_folds)
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.sort(np.concatenate(
            [folds[j] for j in range(n_folds) if j != k]
        ))

        mse_vec[k] = slasso_mse(
            X_train[:, :, train_idx], Y_train[train_idx],
            X_train[:, :, test_idx], Y_train[test_idx],
            alpha_init, beta_init, lam
        )

    return float(mse_vec.mean())
