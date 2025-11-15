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
# # 3 SVD and k-means

# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %load_ext autoreload
# %autoreload 2
from a03_helper import *

# %%
# Cluster the normalized climate data into 5 clusters using k-means and store
# the vector giving the cluster labels for each location.
X_clusters = KMeans(5).fit(X).labels_

# %% [markdown]
# ## 3a

# %%
# Plot the results to the map: use the cluster labels to give the color to each
# point.
plot_xy(lon, lat, X_clusters)
plt.xlabel("Longtitude", fontsize= 16)
plt.ylabel("Latitude", fontsize=16)
# %% [markdown]
# ## 3b

# %%
# YOUR PART HERE
U, s, Vt = svd(X, full_matrices=False)
nextplot()
plot_xy(U[:, 0], U[:, 1], X_clusters)
plt.xlabel("$u_1$", fontsize=16)
plt.ylabel("$u_2$",fontsize=16)

# %% [markdown]
# ## 3c

# %%
# Compute the PCA scores, store in S_PCA (of shape N x Z)
Z = 2
# YOUR PART HERE
U_Z = U[:, :Z]              # first Z left singular vectors
S_Z = np.diag(s[:Z])        # first Z singular values
S_PCA = U_Z @ S_Z           # PCA scores: N x Z

# %%
# cluster and visualize
S_PCA_clusters = KMeans(5).fit(S_PCA).labels_
S_PCA_clusters = match_categories(X_clusters, S_PCA_clusters)

nextplot()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

plot_xy(lon, lat, X_clusters, axis=axs[0])
axs[0].set_title("Original data", fontsize=20)

plot_xy(lon, lat, S_PCA_clusters, axis=axs[1])
axs[1].set_title(f"PCA $(Z={Z})$", fontsize=20)

# global axis labels
fig.supxlabel("Longitude", fontsize=16)
fig.supylabel("Latitude", fontsize=16)

plt.tight_layout()
plt.show()



#%%
#more vis for the report
nextplot()
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ----- Subplot 1: Original clusters on the map -----
plot_xy(lon, lat, X_clusters, axis=axs[0])
axs[0].set_xlabel("Longitude", fontsize=18)
axs[0].set_ylabel("Latitude", fontsize=18)
axs[0].set_title("Original data", fontsize=18)

# ----- Subplot 2: First two left singular vectors -----
plot_xy(U[:, 0], U[:, 1], X_clusters, axis=axs[1])
axs[1].set_xlabel("$u_1$", fontsize=18)
axs[1].set_ylabel("$u_2$", fontsize=18)
axs[1].set_title("Left singular vectors", fontsize=18)

plt.tight_layout()
