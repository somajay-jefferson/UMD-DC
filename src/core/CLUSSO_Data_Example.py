# Author: Jeremy Rubin
# Date: 9/22/23 (original), 2024 (Python port)
# Example code for running CLUSSO and naive approach on a dataset

### Packages required: numpy, scikit-learn, pandas
# pip install numpy scikit-learn pandas

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LassoCV
from sklearn.mixture import GaussianMixture

from SLasso_MSE import lambda_CV_mse
from Mainfunction_albet import Mainfunction_albet

np.random.seed(87543)

######## Generate synthetic data
## Number of subjects
n = 100

# Assume G=2 clusters
G = 2

# Number of initializations for optimization of CLUSSO objective function on real data
M = 5

## Define lower/upper limits for number of objects observed by each subject
low_obj_bound  = 50
high_obj_bound = 100

# Define weights to pre-multiply feature matrices by for generating outcomes
alpha_weights = [0, 1]

# Define mean and variance for noise in observed outcomes
Y_noise_mean = 0
Y_noise_sd   = 5

# Number of folds for cross-validation for average balancing method
n_avg_folds = 5

##### Defining true feature coefficients for purposes of data example, so that
# Xi clearly have an association with the outcome that is determined by these
# coefficients
# Note that these are not observed in real data
beta_star = np.array([5, 5, 0, 0, 2, 2, 2, 2, 0, 10], dtype=float)

# Number of features
q = len(beta_star)

# Grid of regularization parameters to search over for CLUSSO
lambda_grid = np.arange(0.1, 20.1, 0.1)

# Coefficient threshold
beta_thresh = 0.001

### List of design matrices
X_list = [None] * n

# List of scalar outcomes
Y = np.zeros(n)

# Naive average balancing matrix
X_avg = np.zeros((n, q))

## Generate number of objects for each subject
obj_count = np.random.randint(low_obj_bound, high_obj_bound + 1, size=n)

for i in range(n):
    ### Generate Xi for each subject with number of rows equal to number of objects
    # for that subject, and number of columns equal to q/number of features
    Xi = np.random.normal(size=(obj_count[i], q))
    X_list[i] = Xi

    # Take average across objects to get a row of the naive average feature matrix
    X_avg[i, :] = Xi.mean(axis=0)

    # Setting scalar outcomes Y to be feature matrices post-multiplied by feature
    # coefficients and pre-multiplied by random 0/1 weights for each object plus noise
    row_weights = np.random.choice(alpha_weights, size=obj_count[i], replace=True)
    Y[i] = row_weights @ Xi @ beta_star + np.random.normal(Y_noise_mean, Y_noise_sd)

# Row-wise concatenation of all Xi matrices across subjects to jointly cluster
# objects into G clusters
X_all = np.vstack(X_list)

# Assign clustering labels to all subjects using GMM-based clustering
# (equivalent to R's Mclust(X.all, G)$classification)
gmm = GaussianMixture(n_components=G, covariance_type='full',
                      n_init=10, max_iter=1000, tol=1e-5)
gmm.fit(X_all)
clust_label = gmm.predict(X_all) + 1   # 1-indexed to match R convention

# Append cluster labels to the design matrices
clust_label_remaining = clust_label.copy()
for i in range(n):
    pi = obj_count[i]
    X_list[i] = np.column_stack([clust_label_remaining[:pi], X_list[i]])
    clust_label_remaining = clust_label_remaining[pi:]

# Cluster-averaged Xi's to be fit with structured lasso for CLUSSO
clust_X = np.ones((G, q, n))

for i in range(n):
    ## Separate each Xi into two submatrices, one for rows assigned to cluster 1
    # and one assigned to rows assigned to cluster 2
    clust1_X = X_list[i][X_list[i][:, 0] == 1]
    clust2_X = X_list[i][X_list[i][:, 0] == 2]

    # Weights for each cluster are the observed proportion of objects in each cluster
    w1 = clust1_X.shape[0] / X_list[i].shape[0]
    w2 = clust2_X.shape[0] / X_list[i].shape[0]

    ## Average feature values within each cluster and multiply by the appropriate
    ## row-weight  (skip the first label column)
    clust_X[0, :, i] = w1 * clust1_X[:, 1:].mean(axis=0)
    clust_X[1, :, i] = w2 * clust2_X[:, 1:].mean(axis=0)

# Store values of objective function
obj_CLUSSO_vec = np.zeros(M)

# Matrix to store beta hat values for CLUSSO for each of M iterations
beta_matrix  = np.zeros((M, clust_X.shape[1]))
alpha_matrix = np.zeros((M, G))

# Scale outcome variable for CLUSSO fit
Y_CLUSSO = Y - Y.mean()

# Calculate averages for each feature across subjects
X_mean = clust_X.mean(axis=2)   # (G, q)

# Mean-center Xi's
cent_clust_X = clust_X.copy()
for d in range(n):
    cent_clust_X[:, :, d] -= X_mean

for m in range(M):
    # Initialize alpha and beta coefficient vectors for structured lasso - need to be nonzero
    alpha_init = np.random.normal(size=G)
    beta_init  = np.random.normal(size=q)

    # Compute MSEs for grid of regularization/lambda values
    mse_clust_lambda = np.array([
        lambda_CV_mse(clust_X, Y, alpha_init, beta_init, l)
        for l in lambda_grid
    ])

    # Pick the first lambda that gives the lowest MSE
    lambda_clust = lambda_grid[np.argmin(mse_clust_lambda)]

    # Fit CLUSSO model
    CLUSSO_fit = Mainfunction_albet(clust_X, Y, alpha_init, beta_init, lambda_clust)
    alpha_hat  = CLUSSO_fit['alpha']
    beta_hat   = CLUSSO_fit['bet']

    beta_matrix[m, :]  = beta_hat
    alpha_matrix[m, :] = alpha_hat

    # Calculate objective function for CLUSSO to see which beta minimizes the
    # objective function
    obj = sum(
        (Y_CLUSSO[k] - alpha_hat @ cent_clust_X[:, :, k] @ beta_hat) ** 2
        for k in range(n)
    )
    obj += np.sum(np.abs(beta_hat)) * lambda_clust
    obj_CLUSSO_vec[m] = obj

# Pick the beta coefficients that minimize the objective function
best_m    = int(np.argmin(obj_CLUSSO_vec))
beta_hat  = beta_matrix[best_m, :]
alpha_hat = alpha_matrix[best_m, :]

# Naive approach fit: Five-fold cross-validation on the matrix generated by
# Averaging across all objects for each subject
# Equivalent to R's cv.glmnet(X.avg, Y, nfolds=5) with standardize=TRUE
n_subj  = len(Y)
p_feat  = X_avg.shape[1]

X_avg_mu    = X_avg.mean(axis=0)
X_avg_c     = X_avg - X_avg_mu
X_avg_scale = np.sqrt((X_avg_c ** 2).mean(axis=0))
X_avg_scale = np.where(X_avg_scale > 1e-10, X_avg_scale, 1.0)
X_avg_std   = X_avg_c / X_avg_scale

Y_mu = Y.mean()
Y_c  = Y - Y_mu

lambda_max = float(np.max(np.abs(X_avg_std.T @ Y_c)) / n_subj)
eps        = 0.01 if n_subj >= p_feat else 0.0001
lambda_min = max(lambda_max * eps, 1e-10)
lambdas_R  = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), 100))

avg_fit = LassoCV(cv=n_avg_folds, alphas=lambdas_R / 2.0,
                  fit_intercept=False, max_iter=100_000)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    avg_fit.fit(X_avg_std, Y_c)

beta_hat_avg_orig = avg_fit.coef_ / X_avg_scale
intercept         = float(Y_mu - X_avg_mu @ beta_hat_avg_orig)

beta_hat_avg_original = beta_hat_avg_orig.copy()

# Zero out indices corresponding to entries of estimated L1-normalized beta
# coefficient vector that have magnitude less than 0.001
beta_hat_avg = beta_hat_avg_orig / np.sum(np.abs(beta_hat_avg_orig))
zero_inds    = np.abs(beta_hat_avg) < beta_thresh
beta_hat_avg[zero_inds]           = 0.0
beta_hat_avg_original[zero_inds]  = 0.0

# Compute MSE for naive approach
SSE = sum(
    (Y[r] - beta_hat_avg_original @ X_avg[r, :] - intercept) ** 2
    for r in range(n)
)
MSE_avg = SSE / n

## Compute MSE for CLUSSO
SSE_CLUSSO = sum(
    (Y_CLUSSO[b] - alpha_hat @ cent_clust_X[:, :, b] @ beta_hat) ** 2
    for b in range(n)
)
MSE_CLUSSO = SSE_CLUSSO / n

combined_coeff = pd.DataFrame(
    {'CLUSSO': beta_hat, 'Naive approach': beta_hat_avg},
    index=[f'Feature {i+1}' for i in range(q)]
)

# Estimated beta coefficients for both CLUSSO and the naive approach
print(combined_coeff.to_string())

# Estimated MSEs for CLUSSO and the naive approach
print(f"\n[MSE_CLUSSO, MSE_avg] = [{MSE_CLUSSO:.6f}, {MSE_avg:.6f}]")
