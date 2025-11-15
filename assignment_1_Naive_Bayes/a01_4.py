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
# # 4 Model Selection (optional)

# %%
from sklearn.model_selection import KFold
# %load_ext autoreload
# %autoreload 2

import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from a01_helper import *

from a01_functions import nb_train, nb_predict

# %%
# To create folds, you can use:
K = 5
Kf = KFold(n_splits=K, shuffle=True)
for i_train, i_test in Kf.split(X):
    # code here is executed K times, once per test fold
    # i_train has the row indexes of X to be used for training
    # i_test has the row indexes of X to be used for testing
    print(
        "Fold has {:d} training points and {:d} test points".format(
            len(i_train), len(i_test)
        )
    )

# %%
# Use cross-validation to find a good value of alpha. Also plot the obtained
# accuracy estimate (estimated from CV, i.e., without touching test data) as a
# function of alpha.
# YOUR CODE HERE

alphas = [0.1, 0.5, 1, 2, 5, 10, 20]
cv_accuracies = []

K = 5
Kf = KFold(n_splits=K, shuffle=True, random_state=42)

for alpha in alphas:
    fold_accuracies = []
    for i_train, i_test in Kf.split(X):
        X_train, X_val = X[i_train], X[i_test]
        y_train, y_val = y[i_train], y[i_test]

        model = nb_train(X_train, y_train, alpha=alpha)
        preds = nb_predict(model, X_val)

        acc = sklearn.metrics.accuracy_score(y_val, preds["yhat"])
        fold_accuracies.append(acc)

    mean_acc = np.mean(fold_accuracies)
    cv_accuracies.append(mean_acc)
    print(f"α = {alpha:<5} | CV Accuracy = {mean_acc:.4f}")


best_alpha = alphas[np.argmax(cv_accuracies)]
best_acc = max(cv_accuracies)

# %%

plt.figure(figsize=(8, 5))
plt.plot(alphas, cv_accuracies, marker='o', linewidth=2, color="#073E7F")
plt.axvline(best_alpha, color="#29A15C", linestyle="--", label=f"Best α = {best_alpha} (acc={best_acc:.4f})")
plt.xscale("log")
plt.xlabel("Dirichlet Prior α (log scale)")
plt.ylabel("Mean CV Accuracy")
plt.title("Model Selection: Accuracy vs α", fontsize=13, weight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

print(f"\nOptimal α ≈ {best_alpha}  |  Mean CV Accuracy = {best_acc:.4f}")


# %%
# more refined testing for between numbers
# finer grid around the current best alpha=2
alphas_fine = np.arange(1, 3.51, 0.1)
Kf = KFold(n_splits=5, shuffle=True, random_state=42)

means, stds = [], []
for a in alphas_fine:
    fold_acc = []
    for tr, va in Kf.split(X):
        model = nb_train(X[tr], y[tr], alpha=a)
        yhat_va = nb_predict(model, X[va])["yhat"]
        fold_acc.append(sklearn.metrics.accuracy_score(y[va], yhat_va))
    means.append(np.mean(fold_acc))
    stds.append(np.std(fold_acc))

best_idx = np.argmax(means)
best_alpha = alphas_fine[best_idx]
best_mean = means[best_idx]
best_std = stds[best_idx]

print(f"Best α ≈ {best_alpha:.2f} | CV acc = {best_mean:.4f} ± {best_std:.4f}")

plt.figure(figsize=(7,4))
plt.errorbar(alphas_fine, means, yerr=stds, fmt='-o', capsize=3, color="#073E7F", label="CV mean ± std")
plt.scatter(best_alpha, best_mean, color="#29A15C", s=90, zorder=5, label=f"Best α = {best_alpha:.2f}")

plt.axvline(best_alpha, color="#29A15C", linestyle="--", alpha=0.8)
plt.axhline(best_mean, color="#29A15C", linestyle="--", alpha=0.8)

plt.text(best_alpha + 0.05, best_mean + best_std/2, 
         f"α={best_alpha:.2f}\nAcc={best_mean:.4f}±{best_std:.4f}", 
         color="#29A15C", fontsize=10, va='bottom')

plt.xscale('linear')
plt.xlabel("α (Dirichlet prior)")
plt.ylabel("CV Accuracy (mean ± std)")
plt.title("Finer Model Selection", fontsize=13, weight="bold")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()







# %%
#testing with alpha=1.1
model_nb1_1 = nb_train(X, y, alpha=1.1)
pred_nb1_1 = nb_predict(model_nb1_1, Xtest)
yhat = pred_nb1_1["yhat"]#
logprob = pred_nb1_1["logprob"]

# %%
# Accuracy
sklearn.metrics.accuracy_score(ytest, yhat)



# %%
# now let's look at this in more detail
print(sklearn.metrics.classification_report(ytest, yhat))
print(sklearn.metrics.confusion_matrix(ytest, yhat))  