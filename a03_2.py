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

# %% [markdown]
# # 2 The SVD on Weather Data

# %%
import numpy as np
from numpy.linalg import svd as svd
from numpy.linalg import matrix_rank as matrix_rank
import pandas as pd
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from a03_helper import *
np.random.seed(15)  # for reproducibility

# %%
# The date is being loaded via `a03_helper.py`

# Plot the coordinates
plot_xy(lon, lat)


# %% [markdown]
# ## 2a

# %%
# YOUR PART
# Normalize the data to z-scores. Store the result in X.
# For this, complete the code in `a03_helper.py`.

# %%
# Plot histograms of attributes
nextplot()
X.hist(ax=plt.gca())

# %% [markdown]
# ## 2b

# %%
# Compute the SVD of the normalized climate data and store it in variables U,s,Vt. What
# is the rank of the data?
# YOUR PART
U, s, Vt = svd(X, full_matrices=False)

# Print singular values
print("Singular values:")
print(s)

# Compute rank manually (numerical threshold)
rank_manual = np.sum(s > 1e-10)
print(f"Number of non-zero singular values (manual rank): {rank_manual}")

# Compare with numpy's matrix_rank
rank_numpy = matrix_rank(X)
print(f"NumPy matrix_rank: {rank_numpy}")

# %% [markdown]
# ## 2c

# %%
# Here is an example plot.
plot_xy(lon, lat, U[:, 0])

# %%
# For interpretation, it may also help to look at the other component matrices and
# perhaps use other plot functions (e.g., plot_matrix).
# YOUR PART

for k in range(5):
    nextplot()
    plot_xy(lon, lat, U[:, k])
    plt.title(f"Left singular vector u_{k+1}")


# %% [markdown]
# ## 2d

# %%
# Here is an example.
plot_xy(U[:, 0], U[:, 1], lat - np.mean(lat))


# %%
# Plot scatterplots between the first 5 columns of U
# Color once by latitude and once by longitude

for i in range(5):
    for j in range(i + 1, 5):
        nextplot()
        plot_xy(U[:, i], U[:, j], lat - np.mean(lat))
        plt.xlabel(f"U[:, {i}]")
        plt.ylabel(f"U[:, {j}]")
        plt.title(f"U[:,{i}] vs U[:,{j}] – color = latitude (N–S)")

        nextplot()
        plot_xy(U[:, i], U[:, j], lon - np.mean(lon))
        plt.xlabel(f"U[:, {i}]")
        plt.ylabel(f"U[:, {j}]")
        plt.title(f"U[:,{i}] vs U[:,{j}] – color = longitude (E–W)")

# %% [markdown]
# ## 2e

# %%
# 2e(i) Guttman-Kaiser
# YOUR PART

# Singular values larger than 1
Z_gk = np.sum(s > 1)
s_gk = s[s > 1]

print("Singular values greater than 1 (Guttman–Kaiser):")
print(np.round(s_gk, 3))
print(f"\nNumber of components retained: {Z_gk}")


# %%
# 2e(ii) 90% squared Frobenius norm
# YOUR PART

# Compute squared singular values
sing2 = s**2

# Total squared Frobenius norm
total_frob2 = np.sum(sing2)

# Cumulative share of variance explained by first Z singular values
cum_explained = np.cumsum(sing2) / total_frob2

# Find smallest z where at least 90% of variance is explained
Z_90 = np.searchsorted(cum_explained, 0.9) + 1

print("Cumulative explained variance (first 10):", np.round(cum_explained[:10], 3))
print(f"Number of singular values to reach 90% of squared Frobenius norm: {Z_90}")
print(f"Variance explained by first {Z_90} components: {cum_explained[Z_90-1]:.3f}")


# %%
# 2e(iii) Scree test
# YOUR PART
nextplot()
plt.plot(range(1, len(s) + 1), s**2, "o-")
plt.xlabel("Component index")
plt.ylabel(r"$s_k^2$")
plt.title("Scree plot")
plt.grid(True)

for i in range(min(8, len(s))):
    plt.text(i + 1, s[i]**2, f" {i+1}", fontsize=8, color="black", ha="left", va="bottom")

plt.show()

# %%
# 2e(iv) entropy
# YOUR PART
fk = (s**2) / np.sum(s**2)
K = len(fk)
entropy = -np.sum(fk * np.log(fk + 1e-12)) / np.log(K)
cum_fk = np.cumsum(fk)
Z_entropy = np.searchsorted(cum_fk, entropy) + 1

print("Entropy E:", entropy)
print("Entropy-based suggested rank Z:", Z_entropy)
# %%
# 2e(v) random flips
# Random sign matrix: np.random.choice([-1,1], X.shape)
# YOUR PART
sign_matrix = np.random.choice([-1, 1], X.shape)
X_tilde = X * sign_matrix

U, s, Vt = svd(X, full_matrices=False)
m, n = X.shape
min_dim = min(m, n)

ratios = []
Z_values = range(1, min_dim)

for Z in Z_values:
    X_Z = U[:, :Z] @ np.diag(s[:Z]) @ Vt[:Z, :]

    residual_X = X - X_Z

    residual_X_tilde = residual_X * np.random.choice([-1, 1], residual_X.shape)

    norm_XZ = np.linalg.norm(residual_X, 2)
    norm_XZ_tilde = np.linalg.norm(residual_X_tilde, 2)

    ratio = abs(norm_XZ - norm_XZ_tilde) / np.linalg.norm(residual_X, 'fro')
    ratios.append(ratio)

nextplot()
plt.plot(Z_values, ratios, 'o-')
plt.xlabel("Z")
plt.ylabel(r"Relative change")
plt.title("Random sign flip test")
plt.grid(True)

# Identify Z where the relative change is minimal
min_ratio_index = np.argmin(ratios) + 1
print("Smallest relative change observed at Z =", min_ratio_index)


# %% [markdown]
# ## 2f

# %%
# Here is the empty plot that you need to fill (one line per choice of Z: RSME between
# original X and the reconstruction from size-Z SVD of noisy versions)
# YOUR PART
# Define RMSE between X and X_hat
def rmse(X, X_hat):
    """Compute the root-mean-square error (RMSE) between two matrices."""
    N, D = X.shape
    return np.linalg.norm(X - X_hat, 'fro') / np.sqrt(N * D)

# Define noise levels (epsilon values)
epsilons = np.linspace(0, 2, 10)
Z_values = [1, 2, 5, 10, 48]

# Store RMSE results for each Z
rmse_results = {Z: [] for Z in Z_values}

# Compute SVD
U, s, Vt = svd(X, full_matrices=False)

# Loop over noise levels
for eps in epsilons:
    X_noise = X + np.random.randn(*X.shape) * eps

    U_n, s_n, Vt_n = svd(X_noise, full_matrices=False)

    for Z in Z_values:
        X_hat = U_n[:, :Z] @ np.diag(s_n[:Z]) @ Vt_n[:Z, :]
        error = rmse(X, X_hat)
        rmse_results[Z].append(error)
nextplot()
for Z in Z_values:
    plt.plot(epsilons, rmse_results[Z], 'o-', label=f"Z = {Z}")
plt.plot()
plt.xlabel(r"Noise level ($\epsilon$)")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.show()

# %%
