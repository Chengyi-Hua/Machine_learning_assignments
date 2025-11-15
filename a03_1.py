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
# # 1 Intuition on SVD

# %%
import numpy as np
from numpy.linalg import svd as svd
from numpy.linalg import matrix_rank as matrix_rank
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from a03_helper import *
np.random.seed(15)  # for reproducibility

# %%
M1 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

M2 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ]
)

M3 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])

M4 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
)

M5 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]
)

M6 = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
)


# %% [markdown]
# ## 1a

# %%
# Your part

# M1 verification
m1_u = 1/np.sqrt(3) * np.array([1,1,1,0,0])
m1_v = 1/np.sqrt(3) * np.array([1,1,1,0,0])
sigma1 = 3
M1_reconstructed = sigma1 * np.outer(m1_u, m1_v)
print("Original M1:\n", M1)
print("Reconstructed M1 (manual SVD):\n", M1_reconstructed)

# %%

# M2 verification
m2_u = (1/(2*np.sqrt(3))) * np.array([0,2,2,2,0])
m2_v = (2/3) * np.array([0,1,0.5,1,0])
sigma2 = 3*np.sqrt(3)
M2_reconstructed = sigma2 * np.outer(m2_u, m2_v)
print("Original M2:\n", M2)
print("Reconstructed M2 (manual SVD):\n", M2_reconstructed)
# %%

# M3 verification
m3_u = 0.5 * np.array([0,1,1,1,1])
m3_v = 1/np.sqrt(3) * np.array([0,1,1,1])
sigma3 = 2*np.sqrt(3)
M3_reconstructed = sigma3 * np.outer(m3_u, m3_v)
print("Original M3:\n", M3)
print("Reconstructed M3 (manual SVD):\n", M3_reconstructed)

# %%

# M4 verification
m4_u1  = 1/np.sqrt(3) * np.array([1, 1, 1, 0, 0])
m4_v1  = 1/np.sqrt(3) * np.array([1, 1, 1, 0, 0])
m4_s1  = 3  

m4_u2  = 1/np.sqrt(2) * np.array([0, 0, 0, 1, 1])
m4_v2  = 1/np.sqrt(2) * np.array([0, 0, 0, 1, 1])
m4_s2  = 2  

M4_reconstructed = (
    m4_s1 * np.outer(m4_u1, m4_v1) +
    m4_s2 * np.outer(m4_u2, m4_v2)
)

print("Original M4:\n", M4)
print("Reconstructed M4 (manual SVD):\n", M4_reconstructed)
# %% [markdown]
# ## 1b

# %%
# YOUR PART
# Detailed comparison for M1
U, s, Vt = svd(M1)
print(f"Rank(M1): {matrix_rank(M1)}")

print("NumPy U:", np.round(U[:, 0], 4))
print("NumPy Sigma:", np.round(s, 4))
print("NumPy Vt:", np.round(Vt[0, :], 4))

print("\nManual results:")
print("U:", m1_u)
print("Sigma:", sigma1)
print("V:", m1_v)


# %%
# Detailed comparison for M2
U, s, Vt = svd(M2)
print(f"Rank(M2): {matrix_rank(M2)}")

print("NumPy U:", np.round(U[:, 0], 4))
print("NumPy Sigma:", np.round(s, 4))
print("NumPy Vt:", np.round(Vt[0, :], 4))

print("\nManual results:")
print("U:", m2_u)
print("Sigma:", sigma2)
print("V:", m2_v)


# %%
# Detailed comparison for M3
U, s, Vt = svd(M3)
print(f"Rank(M3): {matrix_rank(M3)}")

print("NumPy U:", np.round(U[:, 0], 4))
print("NumPy Sigma:", np.round(s, 4))
print("NumPy Vt:", np.round(Vt[0, :], 4))

print("\nManual results:")
print("U:", m3_u)
print("Sigma:", sigma3)
print("V:", m3_v)


# %%
# Detailed comparison for M4
U, s, Vt = svd(M4)
print(f"Rank(M4): {matrix_rank(M4)}")

print("NumPy U:")
print(np.round(U[:, :2], 4))
print("NumPy Sigma:", np.round(s, 4))
print("NumPy Vt:")
print(np.round(Vt[:2, :], 4))

print("\nManual results:")
print("U:")
print(np.column_stack([m4_u1, m4_u2]))
print("Sigma:", [m4_s1, m4_s2])
print("V:")
print(np.vstack([m4_v1, m4_v2]))


# %%
# Detailed SVD inspection for M5
U5, s5, Vt5 = svd(M5)
print(f"Rank: {matrix_rank(M5)}")
print("Singular values:", np.round(s5, 4))
print("U:")
print(np.round(U5, 4))
print("Vt:")
print(np.round(Vt5, 4))

# %%
# Detailed SVD inspection for M6
U6, s6, Vt6 = svd(M6)
print(f"Rank: {matrix_rank(M6)}")
print("Singular values:", np.round(s6, 4))
print("U:")
print(np.round(U6, 4))
print("Vt:")
print(np.round(Vt6, 4))

# %% [markdown]
# ## 1c

# %%
# You can use the functions svdcomp and plot_matrix from a03_helper.py to visualize the results.
# YOUR PART

matrices = [M1, M2, M3, M4, M5, M6]

for i, M in enumerate(matrices, start=1):
    M_rank1 = svdcomp(M, range(1))

    print(f"Best rank-1 approximation of M{i}:")
    print(M_rank1)
    print()

    # heatmap visualization
    plot_matrix(M_rank1, labels="{:.1f}")

# %% [markdown]
# ## 1d

# %%
# Another method to compute the rank is matrix_rank.
# YOUR PART
U6, s6, Vt6 = svd(M6)

# Print all singular values
print("Singular values of M6 (rounded):", np.round(s6, 8))
print("Singular values of M6:", s6)

# Count non-zero singular values (using numerical threshold)
nonzero_manual = np.sum(s6 > 1e-10)
print(f"Number of non-zero singular values (manual count): {nonzero_manual}")

# Use numpy's matrix_rank for comparison
rank_numpy = matrix_rank(M6)
print(f"Rank of M6 (NumPy matrix_rank): {rank_numpy}")


# %%
