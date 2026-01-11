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
# one file to rule them all
#
# Utility functions for IE 675 Machine Learning, University of Mannheim
#
# Autor: Rainer Gemulla <rgemulla@uni-mannheim.de>
import psutil

import numpy as np
from numpy.linalg import svd
import pandas as pd
import scipy.stats
from scipy.optimize import linear_sum_assignment

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython import get_ipython


# %% [markdown]
# # Misc. Helper Functions
# 
# %%
# Plotting

# setup plotting
# %%
inTerminal = not "IPKernelApp" in get_ipython().config
inJupyterNb = any(
    filter(
        lambda x: x.endswith("jupyter-notebook"), psutil.Process().parent().cmdline()
    )
)
inJupyterLab = any(
    filter(lambda x: x.endswith("jupyter-lab"), psutil.Process().parent().cmdline())
)

get_ipython().run_line_magic(
    "matplotlib", "" if inTerminal else "notebook" if inJupyterNb else "widget"
)


# %%
def nextplot():
    inTerminal = not "IPKernelApp" in get_ipython().config
    if inTerminal:
        plt.clf()  # this clears the current plot
    else:
        plt.figure()  # this creates a new plot


# %%
def plot_matrix(
    M, lim=None, cmap="PiYG", labels="{:.3g}", rownames=None, colnames=None, **kwargs
):
    """Plot the given matrix in a labeled heatmap.

    `M` is the matrix or Pandas data frame.

    `lim` controls the range for the color scale for the entries. If unset, range of
    (-1,+1)*maximum value in `M`. If its a single integer, use range (-1,+1)*lim. Else
    `lim` should be a tuple of (minimum value, maximum value). `cmap` is the colormap
    being used.

    `labels` is a format string used to print the values of each matrix entry. If set to
    `None`, values are not printed (e.g., that's useful for very large matrices).

    `rownames` and `colnames` can be explicitly specified. If they are unset and `M` is
    a Pandas dataframe, row and column names are used from `M`. Otherwise, use indexes.

    """
    if isinstance(M, pd.DataFrame):
        if not colnames:
            colnames = M.columns.values
        if not rownames:
            rownames = M.index.values
        M = M.to_numpy()
    if lim is None:
        lim = np.max(np.abs((M[:, :])))
    if not isinstance(lim, tuple):
        lim = (-lim, lim)
    lim_mean = (lim[0] + lim[1]) / 2.0
    lim_spread = lim[1] - lim_mean

    nextplot()
    plt.matshow(M, fignum=0, cmap=cmap, vmin=lim[0], vmax=lim[1], **kwargs)
    if labels:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(
                    j,
                    i,
                    labels.format(M[i, j]),
                    color="white"
                    if np.abs(M[i, j] - lim_mean) > lim_spread / 2.0
                    else "black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    plt.gca().set_xticks(range(M.shape[1]))
    if colnames is not None:
        plt.gca().set_xticklabels(colnames, rotation="vertical")
    plt.gca().set_yticks(range(M.shape[0]))
    if rownames is not None:
        plt.gca().set_yticklabels(rownames)
    plt.colorbar()


# %%
def plot_cov(M, **kwargs):
    """Plot a covariance matrix.

    `kwargs` are passed on to `plot_matrix`.

    """
    names = M.columns if isinstance(M, pd.DataFrame) else None
    Sigma = np.cov(M.transpose())
    plot_matrix(Sigma, labels=None, rownames=names, colnames=names, **kwargs)


# %%
def plot_xy(x, y, z=None, aspect=1.0, axis=None, **kwargs):
    """Create a scatter plot with colored points.

    `x` and `y` are vectors of coordinates.

    If `z` is `None`, no colors are used. If `z` is a vector of integers (of the same
    length as `x` and `y`), each point `(x[i],y[i])` is colored with color `z[i]`. If
    `z` is a vector of floating point numbers, use a continuous color scale.

    `aspect` sets the aspect ratio of the plot.

    If `axis` is set, put the plot on the specified axis.

    """
    if not axis:
        nextplot()
        axis = plt.gca()
    if z is not None:
        if np.issubdtype(type(z[0]), np.signedinteger):
            # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
            colors = np.array(
                [
                    "#a6cee3",
                    "#1f78b4",
                    "#b2df8a",
                    "#33a02c",
                    "#fb9a99",
                    "#e31a1c",
                    "#fdbf6f",
                    "#ff7f00",
                    "#cab2d6",
                    "#6a3d9a",
                    "#ffff99",
                    "#b15928",
                    "#2f3631",
                ]
            )
            axis.scatter(x, y, c=colors[z % len(colors)], **kwargs)
        else:
            range = np.max(np.abs(z))
            im = axis.scatter(x, y, c=z, cmap="PiYG", vmin=-range, vmax=range, **kwargs)
            plt.colorbar(im, ax=axis)
    else:
        axis.scatter(x, y, **kwargs)
    axis.set_aspect(aspect)


# %%
# Plotting -- MNIST utilities
def showdigit(x):
    "Show one digit as a gray-scale image."
    plt.imshow(x.reshape(28, 28), norm=mpl.colors.Normalize(0, 255), cmap="gray")


# %%
def showdigits(X, y, max_digits=15):
    "Show up to max_digits random digits per class from X with class labels from y."
    num_cols = min(max_digits, max(np.bincount(y)))
    for c in range(10):
        ii = np.where(y == c)[0]
        if len(ii) > max_digits:
            ii = np.random.choice(ii, size=max_digits, replace=False)
        for j in range(num_cols):
            ax = plt.gcf().add_subplot(
                10, num_cols, c * num_cols + j + 1, aspect="equal"
            )
            ax.get_xaxis().set_visible(False)
            if j == 0:
                ax.set_ylabel(c)
                ax.set_yticks([])
            else:
                ax.get_yaxis().set_visible(False)
            if j < len(ii):
                ax.imshow(
                    X[ii[j],].reshape(28, 28),
                    norm=mpl.colors.Normalize(0, 255),
                    cmap="gray",
                )
            else:
                ax.axis("off")


# %%
# Training


def logsumexp(x):
    """Computes log(sum(exp(x)).

    Uses offset trick to reduce risk of numeric over- or underflow. When x is a
    1D ndarray, computes logsumexp of its entries. When x is a 2D ndarray,
    computes logsumexp of each column.

    Keyword arguments:
    x : a 1D or 2D ndarray
    """
    offset = np.max(x, axis=0)
    return offset + np.log(np.sum(np.exp(x - offset), axis=0))


# %%
# Data analysis


def svdcomp(M, components=None):
    """Return sum of the specified components of the SVD of `M``.

    `M` is either a matrix (ndarray) or the SVD of a matrix (3-tuples of U,s,Vt).

    `components` is a list of components to sum up. If unspecified, sum up all the
    components. E.g., if `components=range(z)`, this methods computes the reconstruction
    of the size-z truncated SVD.

    """
    is_matrix = not isinstance(M, tuple)
    if not is_matrix:
        U, s, Vt = M
    if components is None:
        return M if is_matrix else U @ np.diag(s) @ Vt
    else:
        if type(components) == int:
            # this makes sure that shape of factors are retained (e.g., U[:,1] is a
            # vector and U[:,[1]] a matrix with one column)
            components = np.array([components])
        if is_matrix:
            U, s, Vt = svd(M)
        return U[:, components] @ np.diag(s[components]) @ Vt[components, :]


def match_categories(categories1, categories2, return_assignment=False):
    """Match categories of two observation vectors.

    Takes two vectors of categorical observations; both vectors must have the same
    number of observations and the same number of distinct categories. This function
    renames the categories in `categories2` such that the result is as close as possible
    to `categories1` (in Hamming distance).

    This function can be used, for instance, to match two different clusterings. Then
    the categories correspond to cluster numbers and each observation to a data point.

    """

    if len(categories1) != len(categories2):
        raise ValueError("number of instances does not match")
    u1 = np.unique(categories1)
    u2 = np.unique(categories2)
    if len(u1) != len(u2):
        raise ValueError("number of categories does not match")

    C = len(u1)
    dist = np.zeros([C, C])
    for i in range(C):
        pos1 = categories1 == u1[i]
        n1 = np.sum(pos1)
        for j in range(C):
            pos2 = categories2 == u2[j]
            n2 = np.sum(pos2)
            pos12 = pos1 * pos2
            n12 = np.sum(pos12)
            dist[i, j] = (n1 - n12) + (n2 - n12)

    row, col = linear_sum_assignment(dist)
    result = categories1.copy()
    for j in range(C):
        result[categories2 == u2[j]] = u1[row[np.argmax(col == j)]]
    if return_assignment:
        return result, row, col
    else:
        return result


# %% [markdown]
# # Toy Data Generation for Assignment 04

# %% [markdown]
# ## 04 - 1: PPCA


# %%
# You do not need to modify this method.
def ppca_gen(N=10000, D=2, Z=2, sigma2=0.5, mu=None, lambda_=None, Q=None, seed=None):
    """Generate data from a given PPCA model.

    Unless specified otherwise, uses a fixed mean, fixed eigenvalues (variances along
    principal components), and a random orthogonal eigenvectors (principal components).

    """

    # determine model parameters (from arguments or default)
    rng = np.random.RandomState(seed)
    if mu is None:
        mu = np.arange(D) + 1.0
    if Q is None:
        Q = scipy.stats.ortho_group.rvs(D, random_state=rng)
    if lambda_ is None:
        lambda_ = np.arange(D, 0, -1) * 2

    # weight matrix is determined from first Z eigenvectors and eigenvalues of
    # covariance matrix
    Q_Z = Q[:, :Z]
    lambda_Z = lambda_[:Z]
    W = Q_Z * np.sqrt(lambda_Z)  # scales columns

    # generate data
    z = rng.standard_normal(size=(N, Z))  # latent variables
    Eps = rng.standard_normal(size=(N, D)) * np.sqrt(sigma2)  # noise
    X = z @ W.transpose() + mu + Eps  # data points

    # all done
    return dict(
        N=N, D=D, Z=Z, X=X, Zdata=z, mu=mu, Q_Z=Q_Z, lambda_Z=lambda_Z, W=W, Eps=Eps
    )


# %%
# You do not need to modify this method.
def ppca_plot_2d(data, X="X", mu="mu", W="W", alpha=0.05, axis=None, **kwargs):
    """Plot 2D PPCA data along with its weight vectors."""
    if not axis:
        nextplot()
        axis = plt.gca()
    X = data[X] if isinstance(X, str) else X
    plot_xy(X[:, 0], X[:, 1], alpha=alpha, axis=axis, **kwargs)

    # additional plot elements: mean and components
    if mu is not None:
        mu = data[mu] if isinstance(mu, str) else mu
        if W is not None:
            W = data[W] if isinstance(W, str) else W
            head_width = np.linalg.norm(W[:, 0]) / 10.0
            for j in range(W.shape[1]):
                axis.arrow(
                    mu[0],
                    mu[1],
                    W[0, j],
                    W[1, j],
                    length_includes_head=True,
                    head_width=head_width,
                )


# %%

# %% [markdown]
# ## 04 - 2: GMM


# %%
# You do not need to modify this function.
def gmm_gen(N, mu, pi, Sigma=None, seed=None):
    """Generate data from a given GMM model.

    `N` is the number of data points to generate. `mu` and `Sigma` are lists with `K`
    elements holding the mean and covariance matrix of each mixture component. `pi` is a
    `K`-dimensional probability vector of component sizes.

    If `Sigma` is unspecified, a default (random) choice is taken.
    """
    K = len(pi)
    D = len(mu[0])
    rng = np.random.RandomState(seed)
    if Sigma is None:
        Sigma = [
            Q.transpose() @ np.diag([(k + 1) ** 2, k + 1]) @ Q
            for k, Q in enumerate(
                [scipy.stats.ortho_group.rvs(2, random_state=rng) for k in range(K)]
            )
        ]

    components = rng.choice(range(K), p=pi, size=N)
    X = np.zeros([N, D])
    for k in range(K):
        indexes = components == k
        N_k = np.sum(indexes.astype(np.int_))
        if N_k == 0:
            continue

        dist = scipy.stats.multivariate_normal(mean=mu[k], cov=Sigma[k], seed=rng)
        X[indexes, :] = dist.rvs(size=N_k)

    return dict(X=X, components=components, mu=mu, Sigma=Sigma, pi=pi)


# %%
