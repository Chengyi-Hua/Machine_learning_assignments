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
# # 3 Experiments on MNIST Digits Data

# %%
import sklearn, sklearn.metrics
# %load_ext autoreload
# %autoreload 2


from a01_helper import *        
from a01_functions import nb_train, nb_predict

# %%
# Let's train the model on the digits data and predict
model_nb2 = nb_train(X, y, alpha=2)
pred_nb2 = nb_predict(model_nb2, Xtest)
yhat = pred_nb2["yhat"]
logprob = pred_nb2["logprob"]

# %%
# Accuracy
sklearn.metrics.accuracy_score(ytest, yhat)

# %%
# show some digits grouped by prediction; can you spot errors?
nextplot()
showdigits(Xtest, yhat)
plt.suptitle("Digits grouped by predicted label")

# %%
# do the same, but this time show wrong predicitions only
perror = ytest != yhat
nextplot()  
showdigits(Xtest[perror, :], yhat[perror])
plt.suptitle("Errors grouped by predicted label")

# %%
# do the same, but this time on a sample of wrong preditions to see
# error proportions
ierror_s = np.random.choice(np.where(perror)[0], 100, replace=False)
nextplot()
showdigits(Xtest[ierror_s, :], yhat[ierror_s])
plt.suptitle("Errors grouped by predicted label")

# %%
# now let's look at this in more detail
print(sklearn.metrics.classification_report(ytest, yhat))
print(sklearn.metrics.confusion_matrix(ytest, yhat))  # true x predicted

# %%
# plot the confusion matrix
nextplot()
M = sklearn.metrics.confusion_matrix(ytest, yhat)
plt.imshow(M, origin="upper")
for ij, v in np.ndenumerate(M):
    i, j = ij
    plt.text(j, i, str(v), color="white", ha="center", va="center")
plt.xlabel("predicted")
plt.ylabel("true")
plt.colorbar()

# %%
# cumulative accuracy for predictions ordered by confidence (labels show predicted
# confidence)
order = np.argsort(logprob)[::-1]
accuracies = np.cumsum(ytest[order] == yhat[order]) / (np.arange(len(yhat)) + 1)
nextplot()
plt.plot(accuracies)
plt.xlabel("Predictions ordered by confidence")
plt.ylabel("Accuracy")
for x in np.linspace(0.7, 1, 10, endpoint=False):
    index = int(x * (accuracies.size - 1))
    print(np.exp(logprob[order][index]))
    plt.text(index, accuracies[index], "{:.10f}".format(np.exp(logprob[order][index])))





# %% 
# Accuracy for predictions grouped by confidence (labels show
# predicted confidence). Make the plot large (or reduce number of bins) to see
# the labels.
bins = (np.linspace(0, 1, 50) * len(yhat)).astype(int)
mean_accuracy = [
    np.mean(ytest[order][bins[i] : bins[i + 1]] == yhat[order][bins[i] : bins[i + 1]])
    for i in range(len(bins) - 1)
]
nextplot()
plt.bar(np.arange(len(mean_accuracy)), mean_accuracy)
plt.xticks(
    np.arange(len(mean_accuracy)),
    [
        "{:.10f}".format(x)
        for x in np.exp(logprob[order][np.append(bins[1:-1], len(yhat) - 1)])
    ],
)
plt.gcf().autofmt_xdate()
plt.xlabel("Confidence bin")
plt.ylabel("Accuracy")

# %%
#adjusting visuals for storytelling
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

bins = (np.linspace(0, 1, 10) * len(yhat)).astype(int)
mean_accuracy = [
    np.mean(ytest[order][bins[i]:bins[i+1]] == yhat[order][bins[i]:bins[i+1]])
    for i in range(len(bins) - 1)
]
conf_values = np.linspace(0, 1, len(mean_accuracy))

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(
    conf_values, mean_accuracy, width=0.08,
    color=plt.cm.Blues(np.linspace(0.4, 1, len(mean_accuracy))),
    edgecolor="black", alpha=0.9
)

ax.set_title("Prediction Accuracy vs. Confidence Level", fontsize=14, weight="bold")
ax.set_xlabel("Predicted Confidence (normalized)", fontsize=12)
ax.set_ylabel("Mean Accuracy", fontsize=12)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add annotation
for x, y in zip(conf_values, mean_accuracy):
    if y < 0.5:  
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=9, color="gray")

plt.tight_layout()
plt.show()






order = np.argsort(logprob)[::-1]
accuracies = np.cumsum(ytest[order] == yhat[order]) / (np.arange(len(yhat)) + 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(accuracies, color="#073E7F", linewidth=2)
ax.fill_between(np.arange(len(accuracies)), accuracies, color="#99BADF", alpha=0.3)

ax.set_title("Cumulative Prediction Accuracy Ordered by Confidence", fontsize=13, weight="bold")
ax.set_xlabel("Predictions Ordered by Confidence (High → Low)", fontsize=11)
ax.set_ylabel("Cumulative Accuracy", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_ylim(0.8, 1.01)

# Add markers for reference points 
steps = [0.1, 0.5, 1.0]
for s in steps:
    idx = int(s * len(accuracies)) - 1
    ax.plot(idx, accuracies[idx], "o", color="#29A15C")
    ax.text(idx, accuracies[idx] - 0.02, f"{s*100:.0f}% data\n{accuracies[idx]:.3f}", 
            ha="center", va="top", fontsize=9, color="#073E7F")

plt.tight_layout()
plt.show()


import seaborn as sns
import sklearn.metrics as metrics

# Compute confusion matrix
M = metrics.confusion_matrix(ytest, yhat)

plt.figure(figsize=(7, 6))
sns.heatmap(
    M,
    annot=True,
    fmt="d",
    cmap="YlGnBu",       
    cbar_kws={'label': 'Number of Samples'},
    linewidths=0.5,
    linecolor="gray"
)

plt.title("Confusion Matrix – Naive Bayes on MNIST", fontsize=14, weight="bold", pad=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(np.arange(10) + 0.5, np.arange(10), rotation=0)
plt.yticks(np.arange(10) + 0.5, np.arange(10), rotation=0)
plt.tight_layout()
plt.show()


