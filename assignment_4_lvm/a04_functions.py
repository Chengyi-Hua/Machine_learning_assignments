# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from numpy.linalg import svd
import scipy


# %% [markdown]
# ## PPCA


# %%
def ppca_mle(X, Z):
    """Computes the ML estimates of PPCA model parameters.

    Returns a dictionary with keys `mu`, `W`, and `sigma2` and the corresponding ML
    estimates as values.

    """
    N, D = X.shape

    mu_mle = None
    W_mle = None
    sigma2_mle = None

    # Compute the ML estimates of the PPCA model parameters: mu_mle, sigma2_mle (based
    # on mu_mle), and W_mle (based on mu_mle and sigma2_mle). In your code, only use
    # standard matrix/vector operations and svd(...).
    # YOUR CODE HERE

    return dict(mu=mu_mle, W=W_mle, sigma2=sigma2_mle)


# %%
def ppca_nll(X, model):
    """Compute the negative log-likelihood for the given data.

    Model is a dictionary containing keys "mu", "sigma2" and "W" (as produced by
    `ppca_mle` above).

    """
    N, D = X.shape
    
    # YOUR CODE HERE

    mu = model["mu"]
    W = model["W"]
    sigma2 = model["sigma2"]

    C = W @ W.T + sigma2 * np.eye(D)
    sign, logdetC = np.linalg.slogdet(C)
    if sign <= 0:
        raise ValueError("Covariance matrix not positive definite")

    C_inv = np.linalg.inv(C)
    Xc = X - mu

    quad = np.einsum("ni,ij,nj->n", Xc, C_inv, Xc)

    nll = 0.5 * (N * D * np.log(2 * np.pi) + N * logdetC + np.sum(quad))
    return nll



# %% [markdown]
# ## GMM


# %%
def gmm_e(X, model, return_P=False, return_F=False):
    """Perform the E step of EM for a GMM (MLE estimate).

    `model` is a dictionary holding model parameters (keys `mu`, `Sigma`, and `pi`
    defined as in `gmm_gen`).

    Returns a NxK matrix of cluster membership probabilities. If `return_P` is true,
    also returns an NxK matrix holding the density of each data point (row) for each
    component (column).

    """
    return_flag = return_P or return_F
    mu, Sigma, pi = model["mu"], model["Sigma"], model["pi"]
    N, D = X.shape
    K = len(pi)

    P = np.zeros((N, K))
    W = np.zeros((N, K))


    for k in range(K):
        dist = scipy.stats.multivariate_normal(mean=mu[k], cov=Sigma[k])
        densities = dist.pdf(X)
        P[:, k] = pi[k] * densities

    W = P / np.sum(P, axis=1, keepdims=True)

    if return_flag:
        return W, P
    else:
        return W


# %%
def gmm_m(X, W):
    """Perform the M step of EM for a GMM (MLE estimate).

    `W` is the NxK cluster membership matrix computed in the E step. Returns a new model
    (dictionary with keys `mu`, `Sigma`, and `pi` defined as in `gmm_gen`).

    """
    N, D = X.shape
    K = W.shape[1]
    Nk = np.sum(W, axis=0)        
    pi = Nk / N        
    mu = []
    for k in range(K):
        mu_k = np.sum(W[:, k][:, None] * X, axis=0) / Nk[k]
        mu.append(mu_k)

    Sigma = []
    for k in range(K):
        X_centered = X - mu[k]
        Sigma_k = (
            W[:, k][:, None, None] *
            np.matmul(X_centered[:, :, None], X_centered[:, None, :])
        ).sum(axis=0) / Nk[k]

        Sigma.append(Sigma_k)

    return dict(mu=mu, Sigma=Sigma, pi=pi)


# %%
# you do not need to modify this method
def gmm_fit(X, K, max_iter=100, mu0=None, Sigma0=None, pi0=None, gmm_m=gmm_m):
    """Fit a GMM model using EM.

    `K` refers to the number of mixture components to fit. `mu0`, `Sigma0`, and `pi0`
    are initial parameters (automatically set when unspecified).

    """
    N, D = X.shape

    if mu0 is None:
        mu0 = [np.random.randn(D) for k in range(K)]
    if Sigma0 is None:
        Sigma0 = [np.eye(D) * 10 for k in range(K)]
    if pi0 is None:
        pi0 = np.ones(K) / K

    model = dict(mu=mu0, Sigma=Sigma0, pi=pi0)
    for it in range(max_iter):
        W = gmm_e(X, model)
        model = gmm_m(X, W)

    return model
