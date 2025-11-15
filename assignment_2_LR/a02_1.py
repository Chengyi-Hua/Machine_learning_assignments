# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python (ML25)
#     language: python
#     name: ml25
# ---

# %% [markdown]
# # 1. Dataset Statistics

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy

# %load_ext autoreload
# %autoreload 2

from a02_helper import *
from a02_functions import normalize_data

# %%
# look some dataset statistics
scipy.stats.describe(X)

# %%
#see shape
print(X.shape)
# %%
scipy.stats.describe(y)


# %%
print(y.shape)
# %%
# plot the distribution of all features
nextplot()
densities = [scipy.stats.gaussian_kde(X[:, j]) for j in range(D)]
xs = np.linspace(0, np.max(X), 200)
for j in range(D):
    plt.plot(xs, densities[j](xs), label=j)
plt.legend(ncol=5)



# %%
# this plots is not really helpful; go now explore further
# YOUR CODE HERE
# displying histograms of all features
rows = math.ceil(D / 10)
cols = min(10, D)
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 1.5))
axes = axes.flatten()

for j in range(D):
    axes[j].hist(X[:, j], bins=30, density=True, alpha=0.6, color="steelblue")
    axes[j].set_yscale('log')
    axes[j].set_xlim(0, np.percentile(X[:, j], 99))
    axes[j].set_title(features[j], fontsize=8)
    axes[j].tick_params(labelsize=6)

for k in range(D, len(axes)):
    axes[k].axis("off")

plt.suptitle("Histograms of Raw Features (log-scaled y-axis)", y=1.02)
plt.tight_layout()
plt.show()

# %% 
# picking some features have very skewed distributions; consider normalization (for the report)

# Pick features by name
example_features = ["word_freq_free", "word_freq_edu", "capital_run_length_total"]

# Map feature names to column indices
feature_indices = [np.where(features == f)[0][0] for f in example_features]

fig, axes = plt.subplots(1, len(feature_indices), figsize=(12, 4))

for ax, idx in zip(axes, feature_indices):
    ax.hist(X[:, idx], bins=30, density=True, alpha=0.7, color="steelblue")
    ax.set_yscale('log')
    ax.set_xlim(0, np.percentile(X[:, idx], 99))
    ax.set_title(features[idx],    fontsize=20)
    ax.tick_params(labelsize=16)
   

fig.text(-0.00, 0.5, "Density (log scale)", va='center', rotation='vertical', fontsize=20)
fig.text(0.5, 0.01, "Feature Value", ha='center', fontsize=20)
plt.suptitle("Examples of Skewed Raw Feature Distributions", y=1.03)
plt.tight_layout()
plt.show()



# %%
# Let's compute z-scores; create two new variables Xz and Xtestz by completing the
# `normalize` function in `a02_functions.py`. Once you implemented this function, Xz and
# Xtestz will be automatically provided to you in subsequent notebooks.
Xz, Xtestz = normalize_data(X, Xtest)


# %%
# Let's check.
np.mean(Xz, axis=0)  # should be all 0
np.var(Xz, axis=0)  # should be all 1
np.mean(Xtestz, axis=0)  # what do you get here?
np.var(Xtestz, axis=0)

np.sum(Xz**3)  # should be: 1925261.15

# %%
# Explore the normalized data
# YOUR CODE HERE

# KDE plot for normalized features
plt.figure(figsize=(8, 4))
xs = np.linspace(-4, 4, 200) 
densities_z = [scipy.stats.gaussian_kde(Xz[:, j]) for j in range(D)]

for j, d in enumerate(densities_z):
    plt.plot(xs, d(xs), alpha=0.25, lw=1)

plt.title("Kernel Density Plot – Normalized Features", fontsize=20)
plt.xlabel("Feature value (z-score)", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Histogram grid – normalized features (subset for readability)
subset_idx = np.linspace(0, D - 1, 16, dtype=int)  
fig, axes = plt.subplots(4, 4, figsize=(10, 8))
axes = axes.flatten()

for i, j in enumerate(subset_idx):
    axes[i].hist(Xz[:, j], bins=30, density=True, alpha=0.7, color="steelblue")
    axes[i].set_title(features[j])
    axes[i].set_xlim(-4, 4)

for k in range(len(subset_idx), len(axes)):
    axes[k].axis("off")

plt.suptitle("Histograms of Selected Normalized Features", y=1.02)
plt.tight_layout()
plt.show()

# %%
print("Mean (train):", np.round(np.mean(Xz, axis=0)[:5], 3))
print("Std  (train):", np.round(np.std(Xz, axis=0)[:5], 3))
print("Mean (test): ", np.round(np.mean(Xtestz, axis=0)[:5], 3))
print("Std  (test): ", np.round(np.std(Xtestz, axis=0)[:5], 3))
# %%
