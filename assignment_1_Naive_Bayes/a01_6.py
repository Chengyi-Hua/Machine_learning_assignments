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

# %% [markdown]
# # 6 Preprocessing Features and Continuous Naive Bayes (optional)

# %%
import math
import numpy as np
import sklearn, sklearn.metrics
# %load_ext autoreload
# %autoreload 2

from a01_helper import *
from a01_functions import logsumexp, nb_train


# %%
# YOUR CODE HERE
alpha = 1.1 #best from cv in a01_4.py
model_nb = nb_train(X, y, alpha=alpha)
logpriors = model_nb["logpriors"]        # shape (C,)
logcls = model_nb["logcls"]              # shape (C, D, K)
C, D, K = logcls.shape

# %%
# (a) p(y | x_1:D) standard posterior 
def nb_posterior_full(x, model):
    logpriors, logcls = model["logpriors"], model["logcls"]
    C, D, K = logcls.shape
    scores = np.zeros(C)
    for c in range(C):
        scores[c] = logpriors[c] + np.sum(logcls[c, np.arange(D), x])
    lognorm = logsumexp(scores)
    return np.exp(scores - lognorm) 


# (b) p(y | x_1:D') posterior with missing data
def nb_posterior_missing(x, observed_mask, model):
    logpriors, logcls = model["logpriors"], model["logcls"]
    C, D, K = logcls.shape
    scores = np.zeros(C)
    for c in range(C):
        scores[c] = logpriors[c] + np.sum(
            logcls[c, np.where(observed_mask)[0], x[observed_mask]]
        )
    lognorm = logsumexp(scores)
    return np.exp(scores - lognorm)


# (c) p(x_missing | x_observed) predictive sampling
def nb_sample_missing(x, observed_mask, model):
    """Fill in missing features in x by sampling from NB model."""
    post_y = nb_posterior_missing(x, observed_mask, model)
    y_sample = np.random.choice(len(post_y), p=post_y)
    logcls = model["logcls"]
    C, D, K = logcls.shape
    x_filled = x.copy()
    for j in np.where(~observed_mask)[0]:
        probs = np.exp(logcls[y_sample, j, :])
        probs /= probs.sum()
        x_filled[j] = np.random.choice(K, p=probs)
    return x_filled, y_sample

# %%
# alpha=1.1 from CV in task 4
alpha = 1.1
model_nb = nb_train(X, y, alpha=alpha)

# Visualization across mask ratios
mask_ratios = [0.2, 0.4, 0.6, 0.8]
n_examples = 6
np.random.seed(0)
indices = np.random.choice(len(X), n_examples, replace=False)
X_examples = X[indices]

fig, axes = plt.subplots(1, len(mask_ratios), figsize=(2.8 * len(mask_ratios), 4))

for col, mask_ratio in enumerate(mask_ratios):
    imputed_list = []
    for i in range(n_examples):
        x = X_examples[i].copy()
        mask = np.random.rand(x.shape[0]) > mask_ratio
        x_imputed, _ = nb_sample_missing(x, mask, model_nb)
        imputed_list.append(x_imputed)

    X_imputed = np.vstack(imputed_list)
    combined = np.vstack([X_examples, X_imputed])

    side = int(np.sqrt(X.shape[1]))
    n_rows, n_cols = 2, n_examples
    grid = np.ones((side * n_rows, side * n_cols))
    for r in range(n_rows):
        for c in range(n_cols):
            grid[r * side:(r + 1) * side, c * side:(c + 1) * side] = combined[r * n_examples + c].reshape(side, side)

    axes[col].imshow(grid, cmap="gray")
    axes[col].axis("off")
    axes[col].set_title(f"mask={mask_ratio}")

plt.suptitle("Effect of Increasing Missing Data on Naive Bayes Imputation\n(Top: Original, Bottom: Imputed)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()