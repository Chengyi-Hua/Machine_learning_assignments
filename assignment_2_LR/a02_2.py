# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---



# %% [markdown]
# # 2. Maximum Likelihood Estimation

# %% [markdown]
# ## Helper functions
# %%
import numpy as np
import matplotlib as plt

# %load_ext autoreload
# %autoreload 2

from a02_helper import *
from a02_functions import sigma, logsigma, l, dl, gd, sgdepoch, sgd, optimize
# %% [markdown]
# Define the logistic function in `a02_functions.py`. Make sure it operates on both
# scalars and vectors. Then test it below.

# %%
# this should give:
# [0.5, array([0.26894142, 0.5, 0.73105858])]
[sigma(0), sigma(np.array([-1, 0, 1]))]

# %% [markdown]
# Define the logarithm of the logistic function in `a02_functions.py`. Make sure it
# operates on both scalars and vectors.

# %%
# this should give:
# [-0.69314718055994529, array([-1.31326169, -0.69314718, -0.31326169])]
[logsigma(0), logsigma(np.array([-1, 0, 1]))]
#
# %% [markdown]
# ## 2b Log-likelihood and gradient

# %% [markdown]
# Complete the function for the log-likelihood of the logistic regression model in
# `a02_functions.py` and test it below.

# %%
# this should give:
# -47066.641667825766
l(y, Xz, np.linspace(-5, 5, D))

# %% [markdown]
# Now fill out the function that returns the gradient of the log-likelihood of the
# logistic regression model in `a02_functions.py`, and test it below.

# %%
# this should give:
# array([  551.33985842,   143.84116318,   841.83373606,   156.87237578,
#          802.61217579,   795.96202907,   920.69045803,   621.96516752,
#          659.18724769,   470.81259805,   771.32406968,   352.40325626,
#          455.66972482,   234.36600888,   562.45454038,   864.83981264,
#          787.19723703,   649.48042176,   902.6478154 ,   544.00539886,
#         1174.78638035,   120.3598967 ,   839.61141672,   633.30453444,
#         -706.66815087,  -630.2039816 ,  -569.3451386 ,  -527.50996698,
#         -359.53701083,  -476.64334832,  -411.60620464,  -375.11950586,
#         -345.37195689,  -376.22044258,  -407.31761977,  -456.23251936,
#         -596.86960184,  -107.97072355,  -394.82170044,  -229.18125598,
#         -288.46356547,  -362.13402385,  -450.87896465,  -277.03932676,
#         -414.99293368,  -452.28771693,  -167.54649092,  -270.9043748 ,
#         -252.20140951,  -357.72497343,  -259.12468742,   418.35938483,
#          604.54173228,    43.10390907,   152.24258478,   378.16731033,
#          416.12032881])
dl(y, Xz, np.linspace(-5, 5, D))

# %% [markdown]
# ## 2c Gradient descent

# %% [markdown]
# In `a02_functions.py`, define the objective and update function for one
# gradient-descent epoch for fitting an MLE estimate of logistic regression with
# gradient descent (should return a tuple of two functions; see `optimize` in
# `a02_functions.py`). Then test it below.

# %%
# this should give
# [47066.641667825766,
#  array([  4.13777838e+01,  -1.56745627e+01,   5.75882538e+01,
#           1.14225143e+01,   5.54249703e+01,   5.99229049e+01,
#           7.11220141e+01,   4.84761728e+01,   5.78067289e+01,
#           4.54794720e+01,   7.14638492e+01,   1.51369386e+01,
#           3.36375739e+01,   2.15061217e+01,   5.78014255e+01,
#           6.72743066e+01,   7.00829312e+01,   5.29328088e+01,
#           6.16042473e+01,   5.50018510e+01,   8.94624817e+01,
#           2.74784480e+01,   8.51763599e+01,   5.60363965e+01,
#          -2.55865589e+01,  -1.53788213e+01,  -4.67015412e+01,
#          -2.50356570e+00,  -3.85357592e+00,  -2.21819155e+00,
#           3.32098671e+00,   3.86933390e+00,  -2.00309898e+01,
#           3.84684492e+00,  -2.19847927e-01,  -1.29775457e+00,
#          -1.28374302e+01,  -2.78303173e+00,  -5.61671182e+00,
#           1.73657121e+01,  -6.81197570e+00,  -1.20249002e+01,
#           2.65789491e+00,  -1.39557852e+01,  -2.01135653e+01,
#          -2.72134051e+01,  -9.45952961e-01,  -1.02239111e+01,
#           1.52794293e-04,  -5.18938123e-01,  -3.19717561e+00,
#           4.62953437e+01,   7.87893022e+01,   1.88618651e+01,
#           2.85195027e+01,   5.04698358e+01,   6.41240689e+01])
f, update = gd(y, Xz)
[f(np.linspace(-5, 5, D)), update(np.linspace(-5, -5, D), 0.1)]

# %%
# Now you can run gradient descent!
np.random.seed(0)
w0 = np.random.normal(size=D)
wz_gd, vz_gd, ez_gd = optimize(gd(y, Xz), w0, nepochs=500)

# %%
# look at how gradient descent made progess
# YOUR CODE HERE

plt.figure(figsize=(6,4))
plt.plot(vz_gd, label="Negative log-likelihood (GD)")
plt.xlabel("Epoch")
plt.ylabel("Objective value")
plt.title("Gradient Descent Progress")
plt.legend()
plt.show()
# %% [markdown]
# ## 2d Stochastic gradient descent

# %% [markdown]
# Implement stochastic gradient descent in `a02_functions.py`. Then test it with the
# given results below.

# %%
# when you run this multiple times, with 50% probability you should get the
# following result (there is one other result which is very close):
# array([ -3.43689655e+02,  -1.71161311e+02,  -5.71093536e+02,
#         -5.16478220e+01,   4.66294348e+02,  -3.71589878e+02,
#          5.21493183e+02,   1.25699230e+03,   8.33804130e+02,
#          5.63185399e+02,   1.32761302e+03,  -2.64104011e+02,
#          7.10693307e+02,  -1.75497331e+02,  -1.94174427e+02,
#          1.11641507e+02,  -3.30817509e+02,  -3.46754913e+02,
#          8.48722111e+02,  -1.89136304e+02,  -4.25693844e+02,
#         -1.23084189e+02,  -2.95894797e+02,  -2.35789333e+02,
#         -3.38695243e+02,  -3.05642830e+02,  -2.28975383e+02,
#         -2.38075137e+02,  -1.66702530e+02,  -2.27341599e+02,
#         -1.77575620e+02,  -1.49093855e+02,  -1.70028859e+02,
#         -1.50243833e+02,  -1.82986008e+02,  -2.41143708e+02,
#         -3.31047159e+02,  -5.79991185e+01,  -1.98477863e+02,
#         -1.91264948e+02,  -1.17371919e+02,  -1.66953779e+02,
#         -2.01472565e+02,  -1.23330949e+02,  -3.00857740e+02,
#         -1.95853348e+02,  -7.44868073e+01,  -1.11172370e+02,
#         -1.57618226e+02,  -1.25729512e+00,  -1.45536466e+02,
#         -1.43362438e+02,  -3.00429708e+02,  -9.84391082e+01,
#         -4.54152047e+01,  -5.26492232e+01,  -1.45175427e+02])
sgdepoch(y[1:3], Xz[1:3, :], np.linspace(-5, 5, D), 1000)

# %%
# with 50% probability, you should get:
# [40.864973045695081,
#  array([ -3.43689655e+02,  -1.71161311e+02,  -5.71093536e+02,
#          -5.16478220e+01,   4.66294348e+02,  -3.71589878e+02,
#           5.21493183e+02,   1.25699230e+03,   8.33804130e+02,
#           5.63185399e+02,   1.32761302e+03,  -2.64104011e+02,
#           7.10693307e+02,  -1.75497331e+02,  -1.94174427e+02,
#           1.11641507e+02,  -3.30817509e+02,  -3.46754913e+02,
#           8.48722111e+02,  -1.89136304e+02,  -4.25693844e+02,
#          -1.23084189e+02,  -2.95894797e+02,  -2.35789333e+02,
#          -3.38695243e+02,  -3.05642830e+02,  -2.28975383e+02,
#          -2.38075137e+02,  -1.66702530e+02,  -2.27341599e+02,
#          -1.77575620e+02,  -1.49093855e+02,  -1.70028859e+02,
#          -1.50243833e+02,  -1.82986008e+02,  -2.41143708e+02,
#          -3.31047159e+02,  -5.79991185e+01,  -1.98477863e+02,
#          -1.91264948e+02,  -1.17371919e+02,  -1.66953779e+02,
#          -2.01472565e+02,  -1.23330949e+02,  -3.00857740e+02,
#          -1.95853348e+02,  -7.44868073e+01,  -1.11172370e+02,
#          -1.57618226e+02,  -1.25729512e+00,  -1.45536466e+02,
#          -1.43362438e+02,  -3.00429708e+02,  -9.84391082e+01,
#          -4.54152047e+01,  -5.26492232e+01,  -1.45175427e+02])]
f, update = sgd(y[1:3], Xz[1:3, :])
[f(np.linspace(-5, 5, D)), update(np.linspace(-5, 5, D), 1000)]

# %%
# you can run stochastic gradient descent!
wz_sgd, vz_sgd, ez_sgd = optimize(sgd(y, Xz), w0, nepochs=500)

# %% [markdown]
# ## 2e Compare GD and SGD

# %% [markdown]
# Explore the behavior of both methods, given the parameters provided to you!


# %%
# YOUR CODE HERE
# some visualizations to compare GD and SGD

plt.figure(figsize=(9, 4))

#SGD
plt.subplot(1, 2, 1)
plt.plot(vz_sgd, label='SGD', color='blue')
plt.title('Stochastic Gradient Descent', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Negative Log-Likelihood', fontsize=20)
plt.ylim(600, 1000)
plt.tick_params(axis='both', which='major', labelsize=14)

#epoch values to check
epochs = [30, 500]
colors = ['red', 'green']

for e, c in zip(epochs, colors):
    y_val = vz_sgd[e]
    plt.scatter(e, y_val, color=c, zorder=5)
    plt.text(e + 5, y_val + 5, f'({e}, {y_val:.1f})', color=c, fontsize=14)

plt.legend()

#GD
plt.subplot(1, 2, 2)
plt.plot(vz_gd, label='GD', color='orange')
plt.title('Gradient Descent', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(600, 1000)

for e, c in zip(epochs, colors):
    y_val = vz_gd[e]
    plt.scatter(e, y_val, color=c, zorder=5)
    plt.text(e + 5, y_val + 5, f'({e}, {y_val:.1f})', color=c, fontsize=14)

plt.legend(fontsize=14)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(ez_gd, label='GD Step Size', color='orange')
plt.plot(ez_sgd, label='SGD Step Size', color='blue')
plt.title("Step Size Evolution During Optimization")
plt.xlabel("Epochs")
plt.ylabel("Step Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(vz_gd, label="Gradient Descent (GD)", lw=2, color="#073E7F")
plt.plot(vz_sgd, label="Stochastic GD (SGD)", lw=2, alpha=0.8, color="#29A15C")

plt.xlabel("Epochs", fontsize=11)
plt.ylabel("Negative Log-Likelihood", fontsize=11)
plt.title("Convergence of Gradient Descent vs. Stochastic Gradient Descent", fontsize=12, pad=10)

plt.grid(alpha=0.3, linestyle="--")
plt.legend(frameon=True)
plt.tight_layout()
plt.show()

print(f"Final step size (GD):  {ez_gd[-1]:.6f}")
print(f"Final step size (SGD): {ez_sgd[-1]:.6f}")


# some more vis

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# full view on convergence
axes[0].plot(vz_gd, label="Gradient Descent (GD)", lw=2, color="#073E7F")
axes[0].plot(vz_sgd, label="Stochastic GD (SGD)", lw=2, alpha=0.8, color="#29A15C")
axes[0].set_xlabel("Epochs", fontsize=20)
axes[0].set_ylabel("Negative Log-Likelihood", fontsize=20)
axes[0].set_title("Overall Convergence", fontsize=20)
axes[0].grid(alpha=0.3, linestyle="--")
axes[0].legend(frameon=True, fontsize=16)
axes[0].tick_params(axis='both', which='major', labelsize=14)

# -zoom-in on early epochs convergence
axes[1].plot(vz_gd[:50], lw=2, color="#073E7F")
axes[1].plot(vz_sgd[:50], lw=2, alpha=0.8, color="#29A15C")
axes[1].set_xlabel("Epochs (first 50)", fontsize=20)
axes[1].set_title("Zoom-in on Early Convergence", fontsize=20)
axes[1].grid(alpha=0.3, linestyle="--")
axes[1].tick_params(axis='both', which='major', labelsize=16)


plt.suptitle("Comparison of Gradient Descent vs. Stochastic Gradient Descent", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()



# %%

plt.figure(figsize=(9, 4))

#SGD
plt.subplot(1, 2, 1)
plt.plot(vz_sgd, label='SGD', color='blue')
plt.title('Stochastic Gradient Descent', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Negative Log-Likelihood', fontsize=20)
plt.ylim(600, 1000)
plt.tick_params(axis='both', which='major', labelsize=14)

#epoch values to check
epochs = [30, 500]
colors = ['red', 'green']

for e, c in zip(epochs, colors):
    y_val = vz_sgd[e]
    plt.scatter(e, y_val, color=c, zorder=5)
    plt.text(e + 5, y_val + 5, f'({e}, {y_val:.1f})', color=c, fontsize=14)

plt.legend()

#GD
plt.subplot(1, 2, 2)
plt.plot(vz_gd, label='GD', color='orange')
plt.title('Gradient Descent', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(600, 1000)

for e, c in zip(epochs, colors):
    y_val = vz_gd[e]
    plt.scatter(e, y_val, color=c, zorder=5)
    plt.text(e + 5, y_val + 5, f'({e}, {y_val:.1f})', color=c, fontsize=14)

plt.legend(fontsize=14)

plt.tight_layout()
plt.show()
