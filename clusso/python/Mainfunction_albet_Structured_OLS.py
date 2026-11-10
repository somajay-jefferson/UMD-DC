# Port of Supplemental_Code/Mainfunction_albet_Structured_OLS.r
# Structured OLS: the alternating structured lasso with both lasso steps
# replaced by ordinary least squares, and no thresholding. Random CLUSSO uses
# it to refit the features it selected so the selected set has an MSE.

import numpy as np

from K_prdu import K_prdu
from mat_vec_prd import mat_vec_prd


def _ols_slopes(Z, y):
    """
    Equivalent to R: lm(y ~ Z)$coefficients[-1] (intercept fitted, then dropped).

    R's lm returns NA for an aliased column. lstsq returns the minimum-norm
    solution instead, so the two differ only on a rank-deficient design.
    """
    n = Z.shape[0]
    A = np.column_stack([np.ones(n), Z])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs[1:]


def Mainfunction_albet_OLS(X_tr, Y_tr, alpha, bet):
    """
    Estimate alpha and beta by alternating OLS.

    Iteratively:
      1. Fix alpha, solve for beta via OLS on  (alpha' X_i)
      2. Fix beta,  solve for alpha via OLS on (X_i beta)
    until convergence (||beta x alpha - beta0 x alpha0|| < 0.01) or 50 iterations.

    Parameters
    ----------
    X_tr  : (p, q, n) array  -  X_tr[:,:,k] is the k-th subject's matrix
    Y_tr  : (n,)  array
    alpha : (p,)  non-zero initial vector
    bet   : (q,)  non-zero initial vector

    Returns
    -------
    dict {'alpha': (p,), 'bet': (q,)}
    """
    Y_tr  = np.asarray(Y_tr,  dtype=float).flatten()
    alpha = np.asarray(alpha, dtype=float).flatten()
    bet   = np.asarray(bet,   dtype=float).flatten()

    dif  = 100.0
    nrun = 0

    sig_al = np.sign(alpha[np.argmax(np.abs(alpha))])
    alpha  = sig_al * alpha / np.sum(np.abs(alpha))

    nr_alpha = 0.0

    while (dif > 0.01 and nrun < 50
           and np.max(np.abs(alpha)) > 0
           and np.max(np.abs(bet))   > 0):

        nrun  += 1
        bet0   = bet.copy()
        alpha0 = alpha.copy()

        # ── Step 1: fix alpha, update beta via OLS ────────────────────────────
        X_alpha = mat_vec_prd(X_tr, alpha, 'vec_mat')          # (n, q)
        bet     = _ols_slopes(X_alpha, Y_tr)                   # (q,)

        if np.max(np.abs(bet)) > 0:
            sig_bet = np.sign(bet[np.argmax(np.abs(bet))])
            bet     = sig_bet * bet / np.sum(np.abs(bet))

        # ── Step 2: fix beta, update alpha via OLS ────────────────────────────
        X_bet = mat_vec_prd(X_tr, bet, 'mat_vec')              # (n, p)

        if np.max(np.abs(bet)) > 0:
            alpha = _ols_slopes(X_bet, Y_tr)                   # (p,)

        nr_alpha = max(0.0, float(np.sum(np.abs(alpha))))

        if np.max(np.abs(alpha)) > 0:
            sig_al = np.sign(alpha[np.argmax(np.abs(alpha))])
            alpha  = sig_al * alpha / np.sum(np.abs(alpha))

        dif = float(np.linalg.norm(K_prdu(bet, alpha) - K_prdu(bet0, alpha0)))

    # No thresholding: the R version comments it out, OLS fits only.
    alpha = nr_alpha * alpha

    return {'alpha': alpha, 'bet': bet}
