# The Random CLUSSO paper's simulation design, ported from R.
#
# Source: Supplemental_Code.zip -> Random_CLUSSO_non_mainfunctions.R,
# `generate.data` (lines 279-400), with the settings of
# Random_CLUSSO_Data_Analysis.R and Rubin et al. 2026 section 3.1.
#
# It is core's generator (CLUSSO_Functions_Project1_6_16_23.generate_design_
# matrices) with ONE draw changed and different constants:
#
#   latent tubule matrix X_i* (2 x q)   core: independent N(x1, s1), N(x2, s2)
#                                       here: matrix normal, mean rows x1, x2,
#                                       row cov diag(s1, s2), column cov
#                                       blockdiag(I, (1-rho)I + rho J, I) on
#                                       the first 30% / middle 40% / last 30%
#   which features are real             the middle 40% (the correlated block)
#   beta_star                           3 on the middle 40%, 0 elsewhere
#   tubules per subject                 mean 100, variance 5
#   cluster means                       5 and 10
#
# At rho = 0 the column covariance is the identity, and the matrix normal is
# exactly core's independent normals -- same distribution, other constants.
# `check_design()` below tests that and the rho block.
#
# Deliberate differences from the R source, both kept from core's generator:
#   - A subject left with <= 1 tubule in either cluster flags the dataset and
#     it is redrawn (core's no_clust rule). The R version has no such guard.
#   - GMM clustering uses core's sklearn settings, core's stand-in for Mclust.
# Randomness comes from a seeded numpy Generator, so a cohort is a pure
# function of its seed.

import numpy as np
from sklearn.mixture import GaussianMixture

# Random_CLUSSO_Data_Analysis.R lines 24-62 and paper section 3.1
DESIGN = dict(mu=100, sigma_sq_m=5, x1=5.0, x2=10.0, sigma1=1.0, sigma2=3.0,
              sigma_r=1.0, sigma_sq=1.0, beta_value=3.0)


def blocks(q):
    """Feature index ranges: first 30%, middle 40% (real, correlated), last 30%."""
    a = int(round(0.3 * q))
    b = a + int(round(0.4 * q))
    return a, b


def column_cov(q, rho):
    """blockdiag(I_.3q, (1 - rho) I_.4q + rho J_.4q, I_.3q), as in the R code."""
    a, b = blocks(q)
    V = np.eye(q)
    V[a:b, a:b] = rho
    np.fill_diagonal(V, 1.0)
    return V


def tubule_counts(n, mu, sigma_sq_m, rng):
    """Discrete uniform on H integers centred on mu, variance about sigma_sq_m."""
    H = int(np.floor(np.sqrt(12 * sigma_sq_m + 1)))
    if H % 2 == 0:
        H -= 1
    a = int(mu - (H - 1) / 2)
    b = int(mu + (H - 1) / 2)
    return rng.integers(a, b + 1, size=n)


def latent_sampler(q, rho, d=DESIGN):
    """
    The one draw that differs from core: X* = M + A Z B', a 2 x q matrix normal
    with A A' = diag(s1, s2) and B B' = column_cov(q, rho). Returns a function
    of a numpy Generator.
    """
    A = np.diag(np.sqrt([d['sigma1'], d['sigma2']]))
    B = np.linalg.cholesky(column_cov(q, rho))
    M = np.vstack([np.full(q, d['x1']), np.full(q, d['x2'])])
    return lambda rng: M + A @ rng.standard_normal((2, q)) @ B.T


def generate(n, q, rho, seed, d=DESIGN, max_tries=20):
    """
    One cohort in the Random CLUSSO design.

    Returns the same dict as stage1_noise.make_cohort_full, so the stage 4
    pipeline can take either: clust_X (2, q, n), Y (n,), truth (q,),
    clust_acc, X_avg (n, q), X_full (2, q, n), beta_star (q,), alpha_star (2,).
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError('rho must be in [0, 1)')
    rng = np.random.default_rng([seed, 2026])
    a, b = blocks(q)
    beta_star = np.zeros(q)
    beta_star[a:b] = d['beta_value']
    alpha_star = rng.integers(1, 5, size=2).astype(float)
    w = rng.choice(np.round(np.arange(0.2, 0.81, 0.1), 1), size=n, replace=True)

    draw = latent_sampler(q, rho, d)

    for _ in range(max_tries):
        p = tubule_counts(n, d['mu'], d['sigma_sq_m'], rng)
        Y = np.zeros(n)
        X_avg = np.zeros((n, q))
        X_full = np.zeros((2, q, n))
        raw, labels = [], []
        for i in range(n):
            Xs = draw(rng)
            rows = rng.choice(2, size=p[i], replace=True, p=[w[i], 1 - w[i]])
            T = Xs[rows] + rng.normal(0.0, np.sqrt(d['sigma_r']), size=(p[i], q))
            raw.append(T)
            labels.append(rows)
            Xw = Xs * np.array([[w[i]], [1 - w[i]]])
            X_full[:, :, i] = Xw
            Y[i] = alpha_star @ Xw @ beta_star + rng.normal(0.0, np.sqrt(d['sigma_sq']))
            X_avg[i] = T.mean(axis=0)

        X_all = np.vstack(raw)
        truth_lab = np.concatenate(labels)
        gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=10,
                              max_iter=1000, tol=1e-5,
                              random_state=int(rng.integers(0, 2 ** 31 - 1)))
        est = gmm.fit(X_all).predict(X_all)
        acc = max(np.mean(est == truth_lab), np.mean(est != truth_lab))

        clust_X = np.zeros((2, q, n))
        ok = True
        start = 0
        for i in range(n):
            lab = est[start:start + p[i]]
            T = raw[i]
            start += p[i]
            if (lab == 0).sum() <= 1 or (lab == 1).sum() <= 1:
                ok = False
                break
            w1 = (lab == 0).mean()
            clust_X[0, :, i] = w1 * T[lab == 0].mean(axis=0)
            clust_X[1, :, i] = (1 - w1) * T[lab == 1].mean(axis=0)
        if ok:
            return {'clust_X': clust_X, 'Y': Y, 'truth': beta_star != 0,
                    'clust_acc': float(acc), 'X_avg': X_avg, 'X_full': X_full,
                    'beta_star': beta_star, 'alpha_star': alpha_star}
    raise RuntimeError('every draw left some subject with an empty cluster')


def check_design(n=200, q=20, seeds=range(6)):
    """
    Two checks, printed:
      1. rho is real: the latent features in the middle block correlate at
         about rho across subjects, and the outer blocks at about 0.
      2. rho = 0 matches core's generator with Random CLUSSO's constants:
         same latent means/variances, Y spread, tubule counts.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'core'))
    from CLUSSO_Functions_Project1_6_16_23 import generate_design_matrices

    print('1. the latent draw: feature correlation, means and variances (20,000 draws)')
    a, b = blocks(q)
    rng = np.random.default_rng(0)
    for rho in (0.0, 0.3, 0.6, 0.9):
        draw = latent_sampler(q, rho)
        D = np.array([draw(rng) for _ in range(20000)])         # (draws, 2, q)
        for r, lab in ((0, 'cluster 1'), (1, 'cluster 2')):
            C = np.corrcoef(D[:, r, :].T)
            mid = C[a:b, a:b][np.triu_indices(b - a, 1)].mean()
            out = C[:a, :a][np.triu_indices(a, 1)].mean()
            cross = C[:a, a:b].mean()
            print(f'   rho={rho:.1f} {lab}: middle {mid:6.3f}  outer {out:6.3f}  '
                  f'outer-x-middle {cross:6.3f}  mean {D[:, r].mean():6.3f}  '
                  f'var {D[:, r].var(axis=0).mean():5.3f}')
    print('   expect: middle = rho, outer = cross = 0, means 5 / 10, variances 1 / 3')

    print('2. rho = 0 vs core generator with the same constants')
    d = DESIGN
    mine, core = [], []
    for s in seeds:
        c = generate(n, q, 0.0, s)
        np.random.seed(1000 + s)
        w = np.random.choice(np.round(np.arange(0.2, 0.81, 0.1), 1), size=n)
        a_star = c['alpha_star']
        Yc, Xa, cX, Xf, acc, bad = generate_design_matrices(
            n, d['mu'], d['sigma_sq_m'], d['sigma_sq'], d['x1'], d['x2'],
            d['sigma1'], d['sigma2'], d['sigma_r'], a_star, c['beta_star'], w)
        for out, (Y, Xavg) in ((mine, (c['Y'], c['X_avg'])), (core, (Yc, Xa))):
            out.append((Y.mean(), Y.std(), Xavg.mean(), Xavg.std()))
    mine, core = np.array(mine).mean(0), np.array(core).mean(0)
    for name, u, v in zip(('mean Y', 'sd Y', 'mean X_avg', 'sd X_avg'), mine, core):
        print(f'   {name:<11} port {u:9.3f}   core {v:9.3f}')


if __name__ == '__main__':
    check_design()
