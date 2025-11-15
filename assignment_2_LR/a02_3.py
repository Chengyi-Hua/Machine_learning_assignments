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
# # 3 Prediction

# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay,precision_recall_curve

# %load_ext autoreload
# %autoreload 2

from a02_helper import *
from a02_functions import gd, predict, classify, optimize
from a02_functions import sgd

# %%
# Fitted model
# Train both GD and SGD models for comparison
w0 = np.random.normal(size=D)
wz_gd, vz_gd, ez_gd = optimize(gd(y, Xz), w0, nepochs=500, verbose=False)
wz_sgd, vz_sgd, ez_sgd = optimize(sgd(y, Xz), w0, nepochs=500, verbose=False)


# %% [markdown]
# In `a02_functions.py`, complete the `predict` and `classify` methods for the predicted
# spam probability and predicted class label, respectively. Use them to explore your
# previously fitted model.

# %%
# Exploration example: confusion matrix
yhat = predict(Xtestz, wz_gd)
ypred = classify(Xtestz, wz_gd)
print(sklearn.metrics.confusion_matrix(ytest, ypred))  # true x predicted

# %%
# Exploration example: classification report
print(sklearn.metrics.classification_report(ytest, ypred))

# %%
# Exploration Example: precision-recall curve (with annotated thresholds)
nextplot()
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(ytest, yhat)
plt.plot(recall, precision)
for x in np.linspace(0, 1, 10, endpoint=False):
    index = int(x * (precision.size - 1))
    plt.text(recall[index], precision[index], "{:3.2f}".format(thresholds[index]))
plt.xlabel("Recall")
plt.ylabel("Precision")

# %%
# Explore which features are considered important
# YOUR CODE HERE
abs_w = np.abs(wz_gd)
top_idx = np.argsort(abs_w)[-10:][::-1]

print("Top 10 features influencing spam classification:")
for i in top_idx:
    sign = "spam (positive)" if wz_gd[i] > 0 else "spam (negative)"
    print(f"{features[i]:35s}  weight={wz_gd[i]: .3f}  ({sign})")

# Least influential features
least_idx = np.argsort(abs_w)[:10]
print("\nLeast 10 features influencing spam classification:")
for i in least_idx:
    sign = "spam (positive)" if wz_gd[i] > 0 else "spam (negative)"
    print(f"{features[i]:35s}  weight={wz_gd[i]: .3f}  ({sign})")




# %%
# some more visualizations for report comparing and exploring the models that are fitted previously 
# GD and SGD

yhat_gd, ypred_gd = predict(Xtestz, wz_gd), classify(Xtestz, wz_gd)
yhat_sgd, ypred_sgd = predict(Xtestz, wz_sgd), classify(Xtestz, wz_sgd)

# determining Top & least influential features
top_gd, least_gd = np.argsort(np.abs(wz_gd))[-10:][::-1], np.argsort(np.abs(wz_gd))[:10]
top_sgd, least_sgd = np.argsort(np.abs(wz_sgd))[-10:][::-1], np.argsort(np.abs(wz_sgd))[:10]

# GD
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(
    [features[i] for i in top_gd][::-1],
    wz_gd[top_gd][::-1],
    color=["#073E7F" if w > 0 else "#29A15C" for w in wz_gd[top_gd][::-1]]
)
axes[0].set_title("Top 10 Most Influential Features (GD)", fontsize=20)
axes[0].set_xlabel("Weight Value (Importance)", fontsize=20)
axes[0].set_ylabel("Feature", fontsize=20)
axes[0].grid(alpha=0.3, axis="x")
axes[0].tick_params(labelsize=16)

ConfusionMatrixDisplay.from_predictions(
    ytest, ypred_gd, cmap="Blues", colorbar=False,
    text_kw={"fontsize": 20, "fontweight": "bold"}, ax=axes[1]
)
axes[1].set_title("Confusion Matrix (GD)", fontsize=20)
axes[1].set_xlabel("Predicted Label", fontsize=20)
axes[1].set_ylabel("True Label", fontsize=20)
axes[1].tick_params(labelsize=16)

plt.tight_layout()
plt.show()
# %%
# SGD 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(
    [features[i] for i in top_sgd][::-1],
    wz_sgd[top_sgd][::-1],
    color=["#073E7F" if w > 0 else "#29A15C" for w in wz_sgd[top_sgd][::-1]]
)
axes[0].set_title("Top 10 Most Influential Features (SGD)", fontsize=20)
axes[0].set_xlabel("Weight Value (Importance)", fontsize=20)
axes[0].set_ylabel("Feature", fontsize=20)
axes[0].grid(alpha=0.3, axis="x")
axes[0].tick_params(labelsize=16)

ConfusionMatrixDisplay.from_predictions(
    ytest, ypred_sgd, cmap="Greens", colorbar=False,
    text_kw={"fontsize": 20, "fontweight": "bold"}, ax=axes[1]
)
axes[1].set_title("Confusion Matrix (SGD)", fontsize=20)
axes[1].set_xlabel("Predicted Label", fontsize=20)
axes[1].set_ylabel("True Label", fontsize=20)
axes[1].tick_params(labelsize=16)
plt.tight_layout()
plt.show()
#%%
# Least influential features 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(
    [features[i] for i in least_gd][::-1],
    wz_gd[least_gd][::-1],
    color=["#073E7F" if w > 0 else "#29A15C" for w in wz_gd[least_gd][::-1]]
)
axes[0].set_title("10 Least Influential Features (GD)", fontsize=20)
axes[0].set_xlabel("Weight Value (Importance)", fontsize=20)
axes[0].set_ylabel("Feature", fontsize=20)
axes[0].grid(alpha=0.3, axis="x")
axes[0].tick_params(labelsize=16)

axes[1].barh(
    [features[i] for i in least_sgd][::-1],
    wz_sgd[least_sgd][::-1],
    color=["#073E7F" if w > 0 else "#29A15C" for w in wz_sgd[least_sgd][::-1]]
)
axes[1].set_title("10 Least Influential Features (SGD)", fontsize=20)
axes[1].set_xlabel("Weight Value (Importance)", fontsize=20)
axes[1].set_ylabel("Feature", fontsize=20)
axes[1].grid(alpha=0.3, axis="x")
axes[1].tick_params(labelsize=16)
plt.tight_layout()
plt.show()

# %%
# Precision-Recall curves for both models
prec_gd, rec_gd, _ = precision_recall_curve(ytest, yhat_gd)
prec_sgd, rec_sgd, _ = precision_recall_curve(ytest, yhat_sgd)


def annotate_curve(rec, prec, color, k=5, xoff=0.01, yoff=0.01):
    idxs = np.linspace(0, len(rec) - 1, k, dtype=int)
    for i in idxs:
        x = rec[i]
        y = prec[i]
        
        plt.text(x + xoff, y - yoff,
                 f"{x*100:.0f}%",  
                 fontsize=20, color=color)

plt.figure(figsize=(8, 6))
plt.plot(rec_gd, prec_gd, lw=2, label="GD", color="#073E7F")
plt.plot(rec_sgd, prec_sgd, lw=2, linestyle="--", label="SGD", color="#29A15C")


annotate_curve(rec_gd, prec_gd, color="#073E7F", k=5, xoff=0.01, yoff=-0.015)

plt.title("Precision–Recall Curve for GD and SGD", fontsize=20)
plt.xlabel("Recall", fontsize=20)
plt.ylabel("Precision", fontsize=20)
plt.legend(fontsize=20)
plt.grid(alpha=0.3)
plt.tick_params(labelsize=16)
plt.tight_layout()

plt.xlim(0, 1.1)
plt.show()

