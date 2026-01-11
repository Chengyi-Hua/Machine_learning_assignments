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
np.random.seed(15)  
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
print("Mean of normalized attributes (first 5):", np.round(np.mean(X, axis=0)[:5], 3))
print("Std  of normalized attributes (first 5):", np.round(np.std(X, axis=0)[:5], 3))


# %%
# Plot histograms of attributes
nextplot()
plt.figure(figsize=(10, 6))

# Adjustable parameters
title_size = 16
axis_label_size = 20
tick_size = 20
legend_size = 20

groups = {
    "min_temp": [col for col in X.columns if col.startswith("min")],
    "max_temp": [col for col in X.columns if col.startswith("max")],
    "avg_temp": [col for col in X.columns if col.startswith("avg")],
    "rain":     [col for col in X.columns if col.startswith("rain")],
}

for label, cols in groups.items():
    X[cols].stack().hist(
        bins=40,
        alpha=0.5,
        density=True,
        label=label
    )

plt.xlabel("z-score", fontsize=axis_label_size)
plt.ylabel("Density", fontsize=axis_label_size)

plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

plt.legend(fontsize=legend_size)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# %% [markdown]
# ## 2b

# %%
# Compute the SVD of the normalized climate data and store it in variables U,s,Vt. What
# is the rank of the data?
# YOUR PART
U, s, Vt = svd(X, full_matrices=False)
print("Singular values (first 10):")
print(np.round(s[:10], 4))

# Numerical rank using a small threshold
rank_manual = np.sum(s > 1e-10)
print(f"\nManual numerical rank (singular values > 1e-10): {rank_manual}")

# Compare with NumPy's matrix_rank
rank_numpy = matrix_rank(X)
print(f"NumPy matrix_rank: {rank_numpy}")


# %% [markdown]
# ## 2c
# Plot the first five columns of U over longitude/latitude.
for k in range(5):
    plot_xy(lon, lat, U[:, k])
    plt.title(
        f"Left singular vector $u_{{{k+1}}}$ over locations",
        fontsize=16
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)

# %%
# Here is an example plot.
plot_xy(lon, lat, U[:, 0])

# %%
# For interpretation, it may also help to look at the other component matrices and
# perhaps use other plot functions (e.g., plot_matrix).
# YOUR PART
# for interpretation and combining with the SVD background knowledge from the lecture to see the above is potentially enough
# However just some additional tests
# Take first two and last two singular vectors
idx = [0, 1, U.shape[1]-2, U.shape[1]-1]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for ax, k in zip(axs.flatten(), idx):
    plot_xy(lon, lat, U[:, k], axis=ax)
    ax.set_title(f"Left singular vector $u_{{{k+1}}}$ over locations", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)
    ax.tick_params(labelsize=14)

plt.tight_layout()

# %% [markdown]
# ## 2d

# %%
# all comparisons
plot_xy(U[:, 0], U[:, 1], lat - np.mean(lat))
plt.xlabel("$u_1$")
plt.ylabel("$u_2$")
plt.title("$u_1$ vs. $u_2$ (color = latitude)")


# Now systematically: for first five components, color by latitude and longitude
for i in range(5):
    for j in range(i + 1, 5):
        # Color by latitude (North–South)
        plot_xy(U[:, i], U[:, j], lat - np.mean(lat))
        plt.xlabel(f"$u_{i+1}$")
        plt.ylabel(f"$u_{j+1}$")
        plt.title(f"$u_{i+1}$ vs. $u_{j+1}$ (color = latitude)")

        # Color by longitude (East–West)
        plot_xy(U[:, i], U[:, j], lon - np.mean(lon))
        plt.xlabel(f"$u_{i+1}$")
        plt.ylabel(f"$u_{j+1}$")
        plt.title(f"$u_{i+1}$ vs. $u_{j+1}$ (color = longitude)")

# some examples for report
# Color = latitude (North–South)
plot_xy(U[:, 0], U[:, 1], lat - np.mean(lat))
plt.xlabel("$u_1$",fontsize = 16)
plt.ylabel("$u_2$", fontsize = 16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("$u_1$ vs. $u_2$ (color = latitude)", fontsize= 16)
# Color = longitude (East–West)
plot_xy(U[:, 0], U[:, 1], lon - np.mean(lon))
plt.xlabel("$u_1$",fontsize = 16)
plt.ylabel("$u_2$",fontsize = 16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("$u_1$ vs. $u_2$ (color = longitude)",fontsize = 16)


# %% [markdown]
# ## 2e

# %%
# 2e(i) Guttman-Kaiser
# YOUR PART
# Keep components with singular value > 1 (on z-scored data).
Z_gk = np.sum(s > 1.0)
s_gk = s[s > 1.0]

print("Singular values > 1 (Guttman–Kaiser):")
print(np.round(s_gk, 3))
print(f"Suggested rank Z_GK = {Z_gk}")

# %%
# 2e(ii) 90% squared Frobenius norm
# YOUR PART
sing2 = s**2
total_frob2 = np.sum(sing2)

cum_explained = np.cumsum(sing2) / total_frob2
Z_90 = np.searchsorted(cum_explained, 0.9) + 1

print("Cumulative explained variance (first 10):",
      np.round(cum_explained[:10], 3))
print(f"Smallest Z with ≥ 90% of squared Frobenius norm: Z_90 = {Z_90}")
print(f"Variance explained by first {Z_90} components: {cum_explained[Z_90-1]:.3f}")


#%%
# 2e(iii) Scree test
nextplot()
plt.plot(range(1, len(s) + 1), sing2, "o-")
plt.xlabel("Component index k", fontsize=16)
plt.ylabel(r"$s_k^2$", fontsize=16)
plt.yticks(fontsize= 12)
plt.xticks(fontsize= 12)
plt.grid(True)

# Label first few points
for k in range(min(10, len(s))):
    plt.text(k + 1, sing2[k], f" {k+1}", fontsize=12,
             ha="left", va="bottom")


# %%
# 2e(iv) entropy
# YOUR PART
fk = sing2 / total_frob2               # energy fractions
K = len(fk)
entropy = -np.sum(fk * np.log(fk + 1e-12)) / np.log(K)  # normalized entropy
cum_fk = np.cumsum(fk)

# heuristic: choose smallest Z whose cumulative energy exceeds the entropy value
Z_entropy = np.searchsorted(cum_fk, entropy) + 1

print(f"Normalized entropy E = {entropy:.3f}")
print(f"Entropy-based suggested rank Z_entropy = {Z_entropy}")
# %%
# 2e(v) random flips
# Random sign matrix: np.random.choice([-1,1], X.shape)
# YOUR PART
def random_flip_sensitivity(U, s, Vt, Z, ntrials=100):
    """Relative change in X_Z under random sign flips of U[:, :Z]."""
    U_z = U[:, :Z]
    S_z = np.diag(s[:Z])
    Vt_z = Vt[:Z, :]
    X_Z = U_z @ S_z @ Vt_z

    diffs = []
    for _ in range(ntrials):
        signs = np.random.choice([-1, 1], size=U_z.shape)
        U_flip = U_z * signs
        X_Z_flip = U_flip @ S_z @ Vt_z
        diff = np.linalg.norm(X_Z - X_Z_flip, "fro") / np.linalg.norm(X_Z, "fro")
        diffs.append(diff)
    return np.mean(diffs)

m, n = X.shape
max_Z = min(m, n)
Z_grid = range(1, max_Z + 1)

ratios = [random_flip_sensitivity(U, s, Vt, Z) for Z in Z_grid]

nextplot()
plt.plot(list(Z_grid), ratios, "o-")
plt.xlabel("Rank Z", fontsize= 16)
plt.ylabel("Relative change under random sign flips", fontsize = 16)
plt.yticks(fontsize= 12)
plt.xticks(fontsize= 12)
plt.grid(True)

Z_flip = int(Z_grid[np.argmin(ratios)])
print(f"Random flip test – minimal sensitivity at Z_flip = {Z_flip}")
# %% [markdown]
# ## 2f
def rmse(X, X_hat):
    """Root-mean-square error between two matrices."""
    N, D = X.shape
    return np.linalg.norm(X - X_hat, "fro") / np.sqrt(N * D)

# Noise levels
epsilons = np.linspace(0.0, 2.0, 10)
Z_values = [1, 2, 3, 10, 25, 48]  # ranks to compare

# Storage for RMSE curves
rmse_results = {Z: [] for Z in Z_values}
rmse_noise_baseline = []  # RMSE between X and X_noise itself

for eps in epsilons:
    # Add Gaussian noise with std = eps
    X_noise = X + np.random.randn(*X.shape) * eps

    # Baseline: error of noisy data without any denoising
    rmse_noise_baseline.append(rmse(X, X_noise))

    # SVD of noisy data
    U_n, s_n, Vt_n = svd(X_noise, full_matrices=False)

    for Z in Z_values:
        X_hat = U_n[:, :Z] @ np.diag(s_n[:Z]) @ Vt_n[:Z, :]
        err = rmse(X, X_hat)
        rmse_results[Z].append(err)

# Plot RMSE vs epsilon for each Z and baseline
plt.figure(figsize=(10,6))

# Here is the empty plot that you need to fill (one line per choice of Z: RSME between
# original X and the reconstruction from size-Z SVD of noisy versions)
# YOUR PART
# Plot Z-curves first
for Z in Z_values:
    plt.plot(epsilons, rmse_results[Z], "o-", label=f"Z = {Z}")

# Plot baseline last and highlight it
plt.plot(
    epsilons,
    rmse_noise_baseline,
    linestyle="--",
    linewidth=3,
    color="black",
    label="Noisy data (no truncation)"
)

plt.xlabel(r"Noise level ($\epsilon$)",fontsize=16)
plt.ylabel("RMSE vs. original X", fontsize=16)
plt.yticks(fontsize= 12)
plt.xticks(fontsize= 12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()



# %%
# another vis for report

nextplot()
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ----- Plot 1: singular values squared -----
axs[0].plot(range(1, len(s) + 1), sing2, "o-")
axs[0].set_xlabel("Component index k", fontsize=16)
axs[0].set_ylabel(r"$s_k^2$", fontsize=16)
axs[0].tick_params(labelsize=12)
axs[0].grid(True)

for k in range(min(10, len(s))):
    axs[0].text(k + 1, sing2[k], f" {k+1}",
                fontsize=16, ha="left", va="bottom")

# ----- Plot 2: random-flip sensitivity -----
axs[1].plot(list(Z_grid), ratios, "o-")
axs[1].set_xlabel("Rank Z", fontsize=16)
axs[1].set_ylabel("Relative change under random sign flips", fontsize=16)
axs[1].tick_params(labelsize=12)
axs[1].grid(True)

plt.tight_layout()
