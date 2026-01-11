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
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# %load_ext autoreload
# %autoreload 2

from a04_helper import *
from a04_functions import gmm_e, gmm_m, gmm_fit


# %% [markdown]
# # 2 Gaussian Mixture Models

# %% [markdown]
# ## 2a) Toy data

# %%
# Generate a toy dataset and plot it.
toy_gmm = gmm_gen(
    10000,
    [
        np.array([0, 0]),
        np.array([10, 0]),
        np.array([-10, 0]),
        np.array([0, 10]),
        np.array([0, -10]),
    ],
    np.array([0.1, 0.2, 0.25, 0.1, 0.35]),
    seed=4,
)

print(np.sum(toy_gmm["X"] ** 3))  # must be -4380876.753061278

plot_xy(toy_gmm["X"][:, 0], toy_gmm["X"][:, 1], toy_gmm["components"], alpha=0.1)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")





# %% [markdown]
# ## 2b) K-Means

# %%
# Fit 5 clusters using k-means.
kmeans = KMeans(5).fit(toy_gmm["X"])
plot_xy(toy_gmm["X"][:, 0], toy_gmm["X"][:, 1], kmeans.labels_, alpha=0.1)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")


# %% [markdown]
# ## 2c) Fit a GMM

# %%
# Implement the E step of fitting a GMM with MLE by completing gmm_e in a04_functions.py.

# %%
# Test your solution. This should produce:
# (array([[9.99999999e-01, 8.65221693e-10, 1.59675131e-23, 1.14015011e-41, 2.78010004e-65],
# [1.00000000e+00, 3.75078862e-12, 1.55035521e-23, 1.28102693e-34, 1.86750812e-46],
# [9.99931809e-01, 6.81161224e-05, 7.23302032e-08, 2.17176125e-09, 1.62736835e-10]]),
# array([[1.71811600e-08, 5.94620494e-18, 1.82893598e-31, 9.79455071e-50, 1.59217812e-73],
# [1.44159783e-15, 2.16285148e-27, 1.48999246e-38, 9.23362817e-50, 8.97398547e-62],
# [1.85095927e-09, 5.04355064e-14, 8.92595932e-17, 2.01005787e-18, 1.00413265e-19]]))
dummy_model = dict(
    mu=[np.array([k, k + 1]) for k in range(5)],
    Sigma=[np.array([[3, 1], [1, 2]]) / (k + 1) for k in range(5)],
    pi=np.array([0.1, 0.25, 0.15, 0.2, 0.3]),
)
gmm_e(toy_gmm["X"][:3,], dummy_model, return_F=True)

# %%
# Now, implement the M step of fitting a GMM with MLE by completing gmm_e in a04_functions.py.

# %%
# Test your solution. This should produce:
# {'mu': [array([ 6.70641574, -0.47971125]),
#   array([8.2353509 , 2.52134815]),
#   array([-3.0476848 , -1.70722161])],
#  'Sigma': [array([[88.09488652, 11.08635139],
#          [11.08635139,  1.39516823]]),
#   array([[45.82761873, 11.38773232],
#          [11.38773232,  6.87352224]]),
#   array([[98.6662729 , 12.41671355],
#          [12.41671355,  1.56258842]])],
#  'pi': array([0.13333333, 0.53333333, 0.33333333])}
gmm_m(toy_gmm["X"][:3,], np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.0, 1.0, 0.0]]))


# %% [markdown]
# ## 2d+2e) Experiment with GMMs for the toy data


# %%
# Fit on toy data and color each point by most likely component. Also try fitting with 4
# or 6 components.
toy_gmm_fit = gmm_fit(toy_gmm["X"], 5)
# YOUR CODE HERE
W5 = gmm_e(toy_gmm["X"], toy_gmm_fit)

labels_gmm5 = np.argmax(W5, axis=1)

# Plot
plt.figure(figsize=(6, 6))
plot_xy(
    toy_gmm["X"][:, 0],
    toy_gmm["X"][:, 1],
    z=labels_gmm5,
    alpha=0.15
)
plt.title("GMM Clustering (K = 5)", fontsize=14)
plt.tight_layout()
plt.xlabel("x₁", fontsize=14)
plt.ylabel("x₂", fontsize=14)   
plt.show()


# %%
def visualize_gmm_runs(X, K, runs=3):
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 20
    })

    fig, axes = plt.subplots(1, runs, figsize=(5 * runs, 5))
    if runs == 1:
        axes = [axes]

    for i in range(runs):
        model = gmm_fit(X, K)
        W = gmm_e(X, model)
        labels = np.argmax(W, axis=1)

        plot_xy(
            X[:, 0], X[:, 1],
            z=labels, alpha=0.20,
            axis=axes[i]
        )
        axes[i].set_title(f"Run {i+1}", pad=12)
        axes[i].set_xlabel("x₁")
        axes[i].set_ylabel("x₂")
        axes[i].set_aspect('equal')


    fig.suptitle(f"GMM clustering across runs (K={K})", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.show()



visualize_gmm_runs(toy_gmm["X"], K=4, runs=3)
visualize_gmm_runs(toy_gmm["X"], K=6, runs=3)


# %% [markdown]
# ## 2f) Discover the Secret (optional)

# %%
# Load the secret data.
X = np.loadtxt("data/secret_gmm.csv", delimiter=",")


# %%
# How many components are hidden in this data?

