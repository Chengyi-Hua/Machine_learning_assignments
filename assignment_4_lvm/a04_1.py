# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: py3
#     language: python
#     name: py3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

# %load_ext autoreload
# %autoreload 2
from a04_helper import *
from a04_functions import ppca_mle, ppca_nll

# %% [markdown]
# # 1 Probabilistic PCA

# %% [markdown]
# ## 1a) Toy data

# %%
# Generate and plot a toy dataset
toy_ppca = ppca_gen(Z=1, sigma2=0.5, seed=0)
ppca_plot_2d(toy_ppca)
plt.xlabel("x₁", fontsize=14)
plt.ylabel("x₂", fontsize=14)
print(np.sum(toy_ppca["X"] ** 3))  # must be 273244.3990646409


# %%
# Impact of noise
# YOUR CODE HERE
# Impact of noise – nicer visualization in one figure
sigmas = [0.01, 0.1, 0.5, 1.0]

plt.rcParams.update({"font.size": 12})

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, s2 in zip(axes.ravel(), sigmas):
    toy = ppca_gen(Z=1, sigma2=s2, seed=0)
    ppca_plot_2d(toy, axis=ax)
    ax.set_title(rf"$\sigma^2 = {s2}$", fontsize=15)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("x₁", fontsize=14)
    ax.set_ylabel("x₂", fontsize=14)

fig.suptitle("PPCA toy data – impact of noise variance $\sigma^2$", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% [markdown]
# ## 1b) Maximum Likelihood Estimation
def ppca_mle(X, Z):
    N, D = X.shape
    mu_mle = X.mean(axis=0)
    Xc = X - mu_mle
    U, s, Vt = svd(Xc, full_matrices=False)
   
    lambdas = (s ** 2) / N          
    
    if Z < D:
        sigma2_mle = np.mean(lambdas[Z:])   
    else:
        sigma2_mle = 0.0                    
    
    Uz = Vt[:Z, :].T
    
    if sigma2_mle > 0:
        W_mle = Uz @ np.diag(np.sqrt(lambdas[:Z] - sigma2_mle))
    else:
        W_mle = Uz @ np.diag(np.sqrt(lambdas[:Z]))

    return dict(mu=mu_mle, W=W_mle, sigma2=sigma2_mle)

# %%
# Implement MLE for PPCA by completing the function `ppca_mle` in a04_functions.py.

# %%
# Test your solution. This should produce:
# {'mu': array([0.96935329, 1.98309575]),
#  'W': array([[-1.72988776], [-0.95974566]]),
#  'sigma2': 0.4838656103694303}
ppca_mle(toy_ppca["X"], 1)

# %%
# Test your solution. This should produce:
# {'mu': array([0.96935329, 1.98309575]),
# 'W': array([[-1.83371058,  0.33746522], [-1.0173468 , -0.60826214]]),
# 'sigma2': 0.0}
ppca_mle(toy_ppca["X"], 2)

# %% [markdown]
# ## 1c) Negative Log-Likelihood

# %%
# Implement the computation of the conditional negative log-likelihood by completing the function `ppca_nll` in a04_functions.py.

# %%
# Test your solution. This should produce: 32154.198760474777
ppca_nll(toy_ppca["X"], ppca_mle(toy_ppca["X"], 1))


# %% [markdown]
# ## 1d) Discover the Secret!

# %%
# Load the secret data
X = np.loadtxt("data/secret_ppca.csv", delimiter=",")

# %%
# Determine a suitable choice of L using a scree plot.
# Your code here
N, D = X.shape

Xc = X - X.mean(axis=0)
U, s, Vt = svd(Xc, full_matrices=False)
lambdas = (s**2) / N          

plt.figure(figsize=(6,4))
plt.plot(range(1, D+1), lambdas, "o-")
plt.xlabel("Component index")
plt.ylabel("Eigenvalue")
plt.title("Scree plot for secret_ppca")
plt.tight_layout()
plt.show()

# %%
# Determine a suitable choice of Z using validation data.
split = len(X) * 3 // 4
X_train = X[:split,]
X_valid = X[split:,]

# %%
# YOUR CODE HERE

max_Z = min(10, D-1)      
nlls = []

for Z in range(1, max_Z + 1):
    model = ppca_mle(X_train, Z)
    nll = ppca_nll(X_valid, model)
    nlls.append(nll)

plt.figure(figsize=(6,4))
plt.plot(range(1, max_Z + 1), nlls, "o-")
plt.xlabel("Z (latent dimensionality)")
plt.ylabel("Validation NLL")
plt.title("Validation NLL vs latent dimensionality")
plt.tight_layout()
plt.show()

best_Z = np.argmin(nlls) + 1
print("Best Z from validation:", best_Z)

# %%
# some more visualization
N, D = X.shape
Xc = X - X.mean(axis=0)
U, s, Vt = svd(Xc, full_matrices=False)
lambdas = (s**2) / N

split = len(X) * 3 // 4
X_train, X_valid = X[:split, :], X[split:, :]

max_Z = min(10, D-1)
Z_values = np.arange(1, max_Z + 1)
val_nll = []

for Z in Z_values:
    model = ppca_mle(X_train, Z)
    val_nll.append(ppca_nll(X_valid, model))

best_Z = int(Z_values[np.argmin(val_nll)])

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

label_fs = 15
tick_fs = 12

# Scree plot with cumulative variance
ax = axes[0]
ax.plot(range(1, D+1), lambdas, "o-", linewidth=1.8, markersize=5)
ax.set_xlabel("Component index", fontsize=label_fs)
ax.set_ylabel("Eigenvalue", fontsize=label_fs)
ax.tick_params(axis="both", labelsize=tick_fs)
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(range(1, D+1), np.cumsum(lambdas)/np.sum(lambdas), "s--", linewidth=1.8, markersize=5)
ax2.set_ylabel("Cumulative explained variance", fontsize=label_fs)
ax2.tick_params(axis="both", labelsize=tick_fs)

ax.axvline(best_Z, color="grey", linestyle="--", linewidth=1.5)

# Validation NLL
ax = axes[1]
ax.plot(Z_values, val_nll, "o-", linewidth=1.8, markersize=6)
ax.axvline(best_Z, color="grey", linestyle="--", linewidth=1.5)
ax.plot(best_Z, min(val_nll), "ro", markersize=7)  
ax.set_xlabel("Latent dimensionality Z", fontsize=label_fs)
ax.set_ylabel("Validation NLL", fontsize=label_fs)
ax.tick_params(axis="both", labelsize=tick_fs)
ax.grid(True, alpha=0.3)

fig.suptitle("Choosing the latent dimensionality Z in PPCA", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
