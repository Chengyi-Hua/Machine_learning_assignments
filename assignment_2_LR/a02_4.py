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
# # 4 Maximum Aposteriori Estimation

# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
# %load_ext autoreload
# %autoreload 2
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from a02_helper import *
from a02_functions import l, l_l2, dl_l2, gd, gd_l2, logsigma, classify,predict, optimize

# %% [markdown]
# ## 4a Gradient Descent

# %% [markdown]
# Implement the function returning the log-density of the posterior of logistic
# regression, regularized with parameter lambda, in `a02_functions.py`. Then test it
# below.

# %%
# this should give:
# [-47066.641667825766, -47312.623810682911]
[l_l2(y, Xz, np.linspace(-5, 5, D), 0), l_l2(y, Xz, np.linspace(-5, 5, D), 1)]

# %% [markdown]
# Now implement the function to obtain its gradient and test it, in the same manner as
# above.

# %%
# this should give:
# [array([  551.33985842,   143.84116318,   841.83373606,   156.87237578,
#           802.61217579,   795.96202907,   920.69045803,   621.96516752,
#           659.18724769,   470.81259805,   771.32406968,   352.40325626,
#           455.66972482,   234.36600888,   562.45454038,   864.83981264,
#           787.19723703,   649.48042176,   902.6478154 ,   544.00539886,
#          1174.78638035,   120.3598967 ,   839.61141672,   633.30453444,
#          -706.66815087,  -630.2039816 ,  -569.3451386 ,  -527.50996698,
#          -359.53701083,  -476.64334832,  -411.60620464,  -375.11950586,
#          -345.37195689,  -376.22044258,  -407.31761977,  -456.23251936,
#          -596.86960184,  -107.97072355,  -394.82170044,  -229.18125598,
#          -288.46356547,  -362.13402385,  -450.87896465,  -277.03932676,
#          -414.99293368,  -452.28771693,  -167.54649092,  -270.9043748 ,
#          -252.20140951,  -357.72497343,  -259.12468742,   418.35938483,
#           604.54173228,    43.10390907,   152.24258478,   378.16731033,
#           416.12032881]),
#  array([  556.33985842,   148.66259175,   846.4765932 ,   161.33666149,
#           806.89789007,   800.06917193,   924.61902946,   625.71516752,
#           662.75867626,   474.20545519,   774.5383554 ,   355.43897054,
#           458.52686767,   237.04458031,   564.95454038,   867.16124121,
#           789.34009417,   651.44470748,   904.43352968,   545.61254171,
#          1176.21495178,   121.6098967 ,   840.68284529,   634.19739158,
#          -705.95386516,  -629.66826731,  -568.98799574,  -527.33139555,
#          -359.53701083,  -476.82191975,  -411.9633475 ,  -375.65522015,
#          -346.08624261,  -377.11329972,  -408.38904835,  -457.48251936,
#          -598.29817327,  -109.57786641,  -396.60741472,  -231.14554169,
#          -290.60642261,  -364.45545242,  -453.37896465,  -279.71789819,
#          -417.85007654,  -455.32343122,  -170.76077664,  -274.29723194,
#          -255.77283808,  -361.47497343,  -263.05325885,   414.25224198,
#           600.25601799,    38.63962335,   147.59972763,   373.34588176,
#           411.12032881])]
[dl_l2(y, Xz, np.linspace(-5, 5, D), 0), dl_l2(y, Xz, np.linspace(-5, 5, D), 1)]

# %% [markdown]
# Now define the (f,update) tuple handed to the `optimize` function for gradient descent
# on logistic regression with L2 regularization. Then run it below.

# %%
# let's run!
lambda_ = 100
w0 = np.random.normal(size=D)
wz_gd_l2, vz_gd_l2, ez_gd_l2 = optimize(gd_l2(y, Xz, lambda_), w0, nepochs=500)

# %% [markdown]
# ## 4b Effect of Prior

# %%
# YOUR CODE HERE
lambdas = [0, 1, 10, 100, 1000]
results = []

for lam in lambdas:
    w0 = np.zeros(D)
    w_map, v_map, e_map = optimize(gd_l2(y, Xz, lam), w0, nepochs=200, eps0=0.01)
    acc_train = np.mean(classify(Xz, w_map) == y)
    acc_test  = np.mean(classify(Xtestz, w_map) == ytest)
    results.append((lam, acc_train, acc_test, np.linalg.norm(w_map)))
    print(f"λ={lam:<6}  Train Acc={acc_train:.3f}  Test Acc={acc_test:.3f}  ||w||={np.linalg.norm(w_map):.2f}")

# Extract values for plotting
lambdas = [r[0] for r in results]
train_acc = [r[1] for r in results]
test_acc = [r[2] for r in results]
weight_norms = [r[3] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))  

axes[0].semilogx(lambdas, train_acc, "o-", label="Train Acc", linewidth=2)
axes[0].semilogx(lambdas, test_acc, "o-", label="Test Acc", linewidth=2)
axes[0].set_xlabel(r"$\lambda$ (regularization strength, log scale)", fontsize=20)
axes[0].set_ylabel("Accuracy", fontsize=20)
axes[0].set_title("Effect of L2 Regularization", fontsize=20)
axes[0].legend(fontsize=12)
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].tick_params(labelsize=16)

axes[1].semilogx(lambdas, weight_norms, marker='o', color='purple', linewidth=2)
axes[1].set_xlabel(r"$\lambda$ (regularization strength, log scale)", fontsize=20)
axes[1].set_ylabel(r"$\Vert w \Vert_2$ (weight norm)", fontsize=20)
axes[1].set_title("Weight Shrinkage with Regularization", fontsize=20)
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
axes[1].tick_params(labelsize=16)

# Adjust horizontal spacing between plots
plt.subplots_adjust(wspace=2)  

plt.tight_layout()
plt.show()



# %% [markdown]
# ## 4c Composition of Weight Vector




# %%
# YOUR CODE HERE
lambda_list = [10, 50, 100, 500]
weights_by_lambda = []

for lam in lambda_list:
    np.random.seed(0)
    w0 = np.random.normal(size=D)
    w, v, e = optimize(gd_l2(y, Xz, lam), w0, nepochs=500, verbose=False)
    weights_by_lambda.append(np.abs(w))

weights_by_lambda = np.stack(weights_by_lambda, axis=0)
x = np.arange(D)

plt.figure(figsize=(10, 6))

for i, lam in enumerate(lambda_list):
    plt.plot(x, weights_by_lambda[i], marker='.', label=f"λ = {lam}")

plt.xlabel("Feature index", fontsize=20)
plt.ylabel("|Weight| (absolute value)", fontsize=20)
plt.title("Effect of λ on |Weights|", fontsize=20)
plt.legend(fontsize=15)
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()

#%%
# second option





lambdas = [0, 10, 100, 1000]
top_n = 10
w_comparison = {}

for lam in lambdas:
    w0 = np.zeros(D)
    w_map, _, _ = optimize(gd_l2(y, Xz, lam), w0, nepochs=200, eps0=0.01, verbose=False)
    abs_w = np.abs(w_map)
    top_idx = np.argsort(abs_w)[-top_n:][::-1]
    w_comparison[lam] = (w_map[top_idx], [features[i] for i in top_idx])

plt.figure(figsize=(10, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for lam, color in zip(lambdas, colors):
    weights, names = w_comparison[lam]
    plt.plot(range(top_n), weights, "o-", label=f"λ = {lam}", color=color, linewidth=2)

plt.xticks(range(top_n), w_comparison[min(lambdas)][1], rotation=45, ha="right")
plt.xlabel("Feature", fontsize=18)
plt.ylabel("Weight Value", fontsize=18)
plt.title("Top 10 Feature Weights Across Regularization Strengths (GD)", fontsize=20)
plt.legend(fontsize=12)
plt.grid(alpha=0.4, linestyle='--')
plt.tight_layout()

plt.text(7, 1.5, "λ=1000 → weights ≈ 0", color="#d62728", fontsize=12)

plt.show()



# %%
lambda_list = [0.1, 1, 10, 20, 50, 100]

train_nll = []
test_nll  = []
accs      = []
recalls   = []

for lam in lambda_list:
    np.random.seed(0)
    w0 = np.random.normal(size=D)

    w, v, e = optimize(gd_l2(y, Xz, lam), w0, nepochs=500, verbose=False)

    # Compute −logL on train and test
    train_nll.append(-l(y, Xz, w))
    test_nll.append(-l(ytest, Xtestz, w))

    # Predictions on test set for accuracy/recall
    yhat  = predict(Xtestz, w)
    ypred = classify(Xtestz, w)

    accs.append(sklearn.metrics.accuracy_score(ytest, ypred))
    recalls.append(sklearn.metrics.recall_score(ytest, ypred))

# Plot 4 panels
plt.figure(figsize=(7, 10))

plt.subplot(4, 1, 1)
plt.plot(lambda_list, train_nll, marker='o', linewidth=1.5)
plt.title("Effect of λ on Train -log-likelihood")
plt.xlabel("λ (linear scale)")
plt.ylabel("Train −logL")

plt.subplot(4, 1, 2)
plt.plot(lambda_list, test_nll, marker='o', color='tab:orange', linewidth=1.5)
plt.title("Effect of λ on Test -log-likelihood")
plt.xlabel("λ (linear scale)")
plt.ylabel("Test −logL")

plt.subplot(4, 1, 3)
plt.plot(lambda_list, accs, marker='o', color='tab:green', linewidth=1.5)
plt.title("Effect of λ on Accuracy")
plt.xlabel("λ (linear scale)")
plt.ylabel("Accuracy")

plt.subplot(4, 1, 4)
plt.plot(lambda_list, recalls, marker='o', color='tab:red', linewidth=1.5)
plt.title("Effect of λ on Recall")
plt.xlabel("λ (linear scale)")
plt.ylabel("Recall")

plt.tight_layout()
plt.show()



