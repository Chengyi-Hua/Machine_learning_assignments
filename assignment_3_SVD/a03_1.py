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
np.random.seed(15) 
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

M3 = np.array(
    [
        [0, 0, 0, 0],
        [0, 1, 1, 1], 
        [0, 1, 1, 1], 
        [0, 1, 1, 1], 
        [0, 1, 1, 1]
    ]
)

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

#M1 (is rank 1)
u1 = np.array([1,1,1,0,0]) / np.sqrt(3)
v1 = np.array([1,1,1,0,0]) / np.sqrt(3)
s1 = 3
M1_rec = s1 * np.outer(u1, v1)
print("Original M1:\n", M1)
print("\n M1 manual reconstruction:\n", np.round(M1_rec, 3))

#M2 (is rank 1)
u2 = np.array([0,1,1,1,0]) / np.sqrt(3)
# right singular vector
row = np.array([0,2,1,2,0])
v2 = row / np.linalg.norm(row)
# singular value
s2 = np.sqrt(3) * np.linalg.norm(row)   
M2_rec = s2 * np.outer(u2, v2)
print("\n Original M2:\n", M2)
print("\n M2 manual reconstruction:\n", np.round(M2_rec, 3))


#M3 (is rank 1)
u3 = np.array([0,1,1,1,1]) / 2        # norm = 2
v3 = np.array([0,1,1,1]) / np.sqrt(3)
s3 = 2 * np.sqrt(3)
M3_rec = s3 * np.outer(u3, v3)
print("\n Original M3:\n", M3)
print("\n M3 manual reconstruction:\n", np.round(M3_rec, 3))


#M4 (is rank 2)
# component 1
u41 = np.array([1,1,1,0,0]) / np.sqrt(3)
v41 = np.array([1,1,1,0,0]) / np.sqrt(3)
s41 = 3
# component 2
u42 = np.array([0,0,0,1,1]) / np.sqrt(2)
v42 = np.array([0,0,0,1,1]) / np.sqrt(2)
s42 = 2
M4_rec = s41*np.outer(u41, v41) + s42*np.outer(u42, v42)
print("\n Original M4:\n", M4)
print("\n M4 manual reconstruction:\n", np.round(M4_rec, 3))


#M5 and M6

print("\nM5, M6: No manual SVD possible (no clear rank structure identifiable).")
# 

# %% [markdown]
# ## 1b

# %%
# YOUR PART
import numpy as np
from numpy.linalg import svd, matrix_rank

matrices = [M1, M2, M3, M4, M5, M6]
manual_values = {
    1: [3],
    2: [3*np.sqrt(3)],
    3: [2*np.sqrt(3)],
    4: [3, 2],
    5: None,   # no manual solution
    6: None    # no manual solution
}

def match_root_form(x, tol=1e-6):
    for k in range(1, 6):
        for m in range(1, 20):
            if abs(x - k*np.sqrt(m)) < tol:
                return f"{k}√{m}"
    return None

print("SVD comparison (NumPy vs. manual intuition)")
for i, M in enumerate(matrices, start=1):

    U, s, Vt = svd(M)
    s = s[s > 1e-12]          
    approx_symbols = []
    
    for val in s:
        symbolic = match_root_form(val)
        approx_symbols.append(symbolic if symbolic else f"{val:.6f}")

    print(f"\nMatrix M{i}")
    print("NumPy singular values:", np.round(s, 6))
    print("Closed-form match:    ", approx_symbols)
    print("NumPy rank:", matrix_rank(M))

    if manual_values[i] is not None:
        print("Manual expected:      ", [f"{v:.6f}" for v in manual_values[i]])



# %% [markdown]
# ## 1c

# %%
# You can use the functions svdcomp and plot_matrix from a03_helper.py to visualize the results.
# YOUR PART

matrices = [M1, M2, M3, M4, M5, M6]

for i, M in enumerate(matrices, start=1):
    U, s, Vt = svd(M)
    M_rank1 = svdcomp(M, range(1))

    print(f"\nMatrix M{i}")
    print("Top singular value:", np.round(s[0], 4))
    print("Rank-1 approximation:")
    print(np.round(M_rank1, 3))

    if i >= 0:  
        plot_matrix(M_rank1, labels="{:.1f}")


# %% [markdown]
# ## 1d

# %%
# Another method to compute the rank is matrix_rank.
# YOUR PART
U6, s6, Vt6 = svd(M6)

print("Singular values of M6:", np.round(s6, 12))
tol = 1e-10 #standard tolerance
manual_rank = np.sum(s6 > tol)
print("Manual rank (thresholded):", manual_rank)
print("NumPy matrix_rank:", matrix_rank(M6))