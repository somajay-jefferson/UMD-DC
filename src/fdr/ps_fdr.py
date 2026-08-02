# PS-Fdr: false discovery control for penalized variable selection.
#
# Implementation of He, Zhou, Jiang, Wen & Li, "False discovery control for
# penalized variable selections with high-dimensional covariates",
# Stat Appl Genet Mol Biol 17(6), 2018.  Section 2.3-2.4 of that paper is the
# reference for every step below.
#
# The four steps, in the paper's order:
#
#   1. Stability selection on the real data.  B bootstrap resamples, a
#      CV-tuned lasso on each, and the selection frequency Pi_j of every
#      variable across the B fits.
#   2. A permutation-calibrated null.  M permutations of y, stability
#      selection redone on each -- but with the number of selected variables
#      held fixed at the median count from step 1 rather than re-tuned by CV.
#      That fixed count is what keeps the false discovery count from
#      collapsing.
#   3. SAM-style normalization D(u) applied to both arms.
#   4. A step-down cutoff over the ordered statistics, and Fdr-hat = e0-hat/N+.
#
# Requires: numpy, scikit-learn.

import warnings

import numpy as np
from sklearn.linear_model import LassoCV, lars_path


def _standardize(X, y):
    """Centre and scale X columnwise, centre y. Returns (Xs, ys, scale)."""
    Xc = X - X.mean(axis=0)
    scale = np.sqrt((Xc ** 2).mean(axis=0))
    scale = np.where(scale > 1e-10, scale, 1.0)
    return Xc / scale, y - y.mean(), scale


def lasso_support_cv(X, y, n_folds=10, random_state=None):
    """
    Support of a cross-validated lasso fit -- the selection rule the paper
    treats as the status quo, and the rule used on the real-data arm.

    Returns a boolean mask of length p.
    """
    Xs, ys, _ = _standardize(X, y)
    fit = LassoCV(cv=n_folds, fit_intercept=False, max_iter=100_000,
                  random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit.fit(Xs, ys)
    return fit.coef_ != 0.0


def lasso_support_fixed_k(X, y, k):
    """
    Support of size ``k``: walk the lasso path down in lambda and stop at the
    first point where k variables are active.

    This is the "do not re-run CV" rule of step 2.  Walking the path to a
    target size is what a lasso makes cheap, and it is the reason the permuted
    arm is comparable to the real arm rather than systematically emptier.

    Note this is the support *at* the lambda where the path first reaches size
    k, not the first k variables to enter it.  The two differ whenever a
    variable enters, drops out and re-enters, which the lasso modification of
    LARS permits; the support at a single lambda is the one a practitioner
    would report.  Where the path steps straight past k, the k largest
    coefficients at the first step above k are kept.

    Returns a boolean mask of length p with at most k true entries (fewer only
    if the path ends before reaching size k).
    """
    p = X.shape[1]
    mask = np.zeros(p, dtype=bool)
    if k <= 0:
        return mask

    Xs, ys, _ = _standardize(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, coefs = lars_path(Xs, ys, method='lasso', return_path=True)

    nnz = (coefs != 0).sum(axis=0)
    reached = np.flatnonzero(nnz >= k)
    step = int(reached[0]) if reached.size else int(np.argmax(nnz))

    beta = coefs[:, step]
    support = np.flatnonzero(beta)
    if support.size > k:
        support = support[np.argsort(-np.abs(beta[support]))[:k]]

    mask[support] = True
    return mask


def stability_selection(X, y, B=50, k_fixed=None, rng=None, n_folds=10,
                        return_masks=False):
    """
    Step 1 (and, with ``k_fixed`` set, the inner loop of step 2).

    Resample n rows with replacement B times, fit a lasso on each resample,
    and record how often each variable is selected.

    Parameters
    ----------
    X, y    : (n, p) design and (n,) outcome
    B       : number of bootstrap resamples
    k_fixed : if None, each fit picks its own support by cross-validation;
              otherwise every fit returns exactly this many variables
    rng     : numpy Generator
    return_masks : also return the (B, p) selection indicator matrix that the
              frequencies are counted from

    Returns
    -------
    Pi     : (p,) selection frequency in [0, 1]
    counts : (B,) support size of each bootstrap fit
    masks  : (B, p) bool, only when ``return_masks``
    """
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[0]

    hits = np.zeros(X.shape[1])
    counts = np.zeros(B, dtype=int)
    masks = np.zeros((B, X.shape[1]), dtype=bool)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]

        if k_fixed is None:
            mask = lasso_support_cv(Xb, yb, n_folds=n_folds,
                                    random_state=int(rng.integers(0, 2**31 - 1)))
        else:
            mask = lasso_support_fixed_k(Xb, yb, k_fixed)

        hits += mask
        masks[b] = mask
        counts[b] = int(mask.sum())

    if return_masks:
        return hits / B, counts, masks
    return hits / B, counts


def sam_normalize(u, nu):
    """
    Step 3.  D(u) = u / (sqrt(u(1-u)) + nu), from Tusher et al.'s Significance
    Analysis of Microarrays, with nu = 1/B.

    Strictly increasing on [0, 1], so thresholding Z is the same as
    thresholding Pi -- the normalization spreads the scale apart, it does not
    reorder anything.
    """
    u = np.asarray(u, dtype=float)
    return u / (np.sqrt(np.clip(u * (1.0 - u), 0.0, None)) + nu)


def ps_fdr(X, y, q=0.1, B=50, M=100, seed=None, n_folds=10, null_mode='fixed'):
    """
    Run the whole procedure and return every intermediate quantity.

    Parameters
    ----------
    X, y : (n, p) design and (n,) outcome
    q    : target false discovery level
    B    : bootstrap resamples per stability-selection pass
    M    : permutations used to calibrate the null
    seed : integer seed for the bootstrap and permutation draws
    null_mode : 'fixed' follows the paper and holds the permuted-arm support
              size at the real arm's median; 'cv' lets cross-validation choose
              it freely on permuted data.  'cv' is the variant the paper argues
              is invalid, kept here so the difference can be measured.

    Returns
    -------
    dict with keys

      Pi           (p,)    real-arm selection frequency, in variable order
      counts       (B,)    real-arm support sizes
      masks        (B, p)  real-arm selection indicators, one row per resample
      k            int     median support size -- the count fixed on the null arm
      Pi_null      (M, p)  permuted-arm frequencies, each row sorted ascending
      Pi_bar       (p,)    mean of Pi_null over permutations, by rank
      Z, Z_bar     (p,)    normalized real and null order statistics
      Z_null       (M, p)  normalized permuted order statistics
      sweep        list    one dict per candidate Delta: delta, cutoff, n_pos,
                           e0, fdr, selected (variable indices)
      selected             chosen variable indices at level q, most stable first
      fdr_hat      float   the Fdr estimate at the chosen cutoff
      delta        float   the chosen Delta
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p = X.shape[1]
    nu = 1.0 / B

    rng = np.random.default_rng(seed)

    # --- step 1: stability selection on the real data ---------------------
    Pi, counts, masks = stability_selection(X, y, B=B, rng=rng,
                                            n_folds=n_folds,
                                            return_masks=True)

    # The number of variables the null arm is forced to select.  Median, not
    # mean: the paper wants a typical CV-tuned support size, and the bootstrap
    # support-size distribution is skewed.
    k = int(np.median(counts))

    # --- step 2: permutation-calibrated null ------------------------------
    k_null = k if null_mode == 'fixed' else None

    Pi_null = np.zeros((M, p))
    for m in range(M):
        y_perm = rng.permutation(y)
        Pi_m, _ = stability_selection(X, y_perm, B=B, k_fixed=k_null, rng=rng,
                                      n_folds=n_folds)
        # Sorted ascending: the null is a distribution over *ranks*, not over
        # variable identities.  Pi_(j), never Pi_j.
        Pi_null[m] = np.sort(Pi_m)

    Pi_bar = Pi_null.mean(axis=0)

    # --- step 3: normalize both arms --------------------------------------
    order = np.argsort(Pi, kind='stable')      # ascending, as in the paper
    Pi_ordered = Pi[order]

    Z = sam_normalize(Pi_ordered, nu)
    Z_bar = sam_normalize(Pi_bar, nu)
    Z_null = sam_normalize(Pi_null, nu)

    # --- step 4: step down, cut, estimate ---------------------------------
    gaps = Z - Z_bar
    candidates = np.unique(np.concatenate([[0.0], gaps[gaps > 0]]))

    sweep = []
    for delta in candidates:
        clears = Z >= Z_bar + delta
        if not clears.any():
            continue
        cutoff = float(Z[clears].min())

        n_pos = int((Z >= cutoff).sum())
        e0 = float((Z_null >= cutoff).sum()) / M
        fdr = e0 / n_pos if n_pos > 0 else np.nan

        # variables at or above the cutoff, most stable first
        picked = order[Z >= cutoff][::-1]

        sweep.append({'delta': float(delta), 'cutoff': cutoff,
                      'n_pos': n_pos, 'e0': e0, 'fdr': float(fdr),
                      'selected': picked.tolist()})

    # Largest set whose estimated Fdr clears the target level.
    ok = [s for s in sweep if s['fdr'] <= q]
    best = max(ok, key=lambda s: s['n_pos']) if ok else None

    return {
        'Pi': Pi, 'counts': counts, 'k': k, 'masks': masks,
        'Pi_null': Pi_null, 'Pi_bar': Pi_bar,
        'order': order,
        'Z': Z, 'Z_bar': Z_bar, 'Z_null': Z_null,
        'sweep': sweep,
        'selected': np.array(best['selected'] if best else [], dtype=int),
        'fdr_hat': best['fdr'] if best else np.nan,
        'delta': best['delta'] if best else np.nan,
    }


def fdp_power(selected, truth_mask):
    """
    Realized false discovery proportion and power of a selected set.

    FDP is 0 by convention when nothing is selected, matching the paper's
    reporting.
    """
    selected = np.asarray(selected, dtype=int)
    truth_mask = np.asarray(truth_mask, dtype=bool)

    n_sel = selected.size
    n_true = int(truth_mask.sum())
    n_false = int((~truth_mask[selected]).sum()) if n_sel else 0

    fdp = n_false / n_sel if n_sel else 0.0
    power = int(truth_mask[selected].sum()) / n_true if n_true else np.nan
    return fdp, power
