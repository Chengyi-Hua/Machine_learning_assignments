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
# # 5 Exploration (optional)

# %%
import math,itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional
import torch
import torch.nn as nn
import torch.utils.data
import sklearn
from sklearn.model_selection import KFold
from a02_helper import *
from a02_functions import normalize_data, gd_l2, optimize, classify
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %load_ext autoreload
# %autoreload 2

from a02_helper import *


# %% [markdown]
# ### 5 Exploration: PyTorch

# %%
# if you want to experiment, here is an implementation of logistic
# regression in PyTorch

# prepare the data
Xztorch = torch.FloatTensor(Xz)
ytorch = torch.LongTensor(y)
train = torch.utils.data.TensorDataset(Xztorch, ytorch)


# manual implementation of logistic regression (without bias)
class LogisticRegression(torch.nn.Module):
    def __init__(self, D, C):
        super(LogisticRegression, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.randn(D, C) / math.sqrt(D)
        )  # xavier initialization
        self.register_parameter("W", self.weights)

    def forward(self, x):
        out = torch.matmul(x, self.weights)
        out = torch.nn.functional.log_softmax(out)
        return out


# define the objective and update function. here we ignore the learning rates and
# parameters given to us by optimize (they are stored in the PyTorch model and
# optimizer, resp., instead)
def opt_pytorch():
    model = LogisticRegression(D, 2)
    criterion = torch.nn.NLLLoss(reduction="sum")
    # change the next line to try different optimizers
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def objective(_):
        outputs = model(Xztorch)
        return criterion(outputs, ytorch)

    def update(_1, _2):
        for i, (examples, labels) in enumerate(train_loader):
            outputs = model(examples)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        W = model.state_dict()["W"]
        w = W[:, 1] - W[:, 0]
        return w

    return (objective, update)


# %%
# run the optimizer
learning_rate = 0.01
batch_size = 100  # number of data points to sample for gradient estimate
shuffle = True  # sample with replacement (false) or without replacement (true)

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
wz_t, vz_t, _ = optimize(opt_pytorch(), None, nepochs=100, eps0=None, verbose=True)




# %%

# Compare GD on normalized vs raw data
Xz, Xtestz = normalize_data(X, Xtest)
def run_map(Xtrain, ytrain, Xtest, ytest, lam=10):
    w0 = np.zeros(Xtrain.shape[1])
    w_map, _, _ = optimize(gd_l2(ytrain, Xtrain, lam), w0, nepochs=200, eps0=0.01, verbose=False)
    acc_train = np.mean(classify(Xtrain, w_map) == ytrain)
    acc_test  = np.mean(classify(Xtest, w_map) == ytest)
    return acc_train, acc_test

acc_norm = run_map(Xz, y, Xtestz, ytest, lam=10)
acc_raw  = run_map(X,  y, Xtest,  ytest, lam=10)

print(f"Normalized data accuracy: Train={acc_norm[0]:.3f}, Test={acc_norm[1]:.3f}")
print(f"Raw data accuracy:        Train={acc_raw[0]:.3f}, Test={acc_raw[1]:.3f}")


# Add a bias feature (not scaled)
Xb = np.column_stack([np.ones(len(X)), X])
Xb_test = np.column_stack([np.ones(len(Xtest)), Xtest])
acc_bias = run_map(Xb, y, Xb_test, ytest, lam=10)
print(f"With bias feature: Train={acc_bias[0]:.3f}, Test={acc_bias[1]:.3f}")


# Reduce training size + Cross-validation (effect on MAP)
def cross_validate_map(X, y, lam=10, k=5, nepochs=200):
    """Performs k-fold cross-validation for MAP logistic regression."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accs_train, accs_val, norms = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        w0 = np.zeros(X.shape[1])
        w_map, _, _ = optimize(gd_l2(y[train_idx], X[train_idx], lam), w0, nepochs=nepochs, eps0=0.01, verbose=False)
        acc_train = np.mean(classify(X[train_idx], w_map) == y[train_idx])
        acc_val   = np.mean(classify(X[val_idx], w_map) == y[val_idx])
        accs_train.append(acc_train)
        accs_val.append(acc_val)
        norms.append(np.linalg.norm(w_map))
        print(f"Fold {fold+1}: Train={acc_train:.3f}, Val={acc_val:.3f}, ||w||={norms[-1]:.2f}")
    print(f"\n{k}-fold CV summary (λ={lam}): "
          f"Train mean={np.mean(accs_train):.3f}±{np.std(accs_train):.3f}, "
          f"Val mean={np.mean(accs_val):.3f}±{np.std(accs_val):.3f}, "
          f"||w|| mean={np.mean(norms):.2f}")
    return np.mean(accs_train), np.mean(accs_val), np.mean(norms)

# Reduced training set (30%)
frac = 0.3
N_sub = int(frac * len(Xz))
idx = np.random.choice(len(Xz), N_sub, replace=False)
acc_small = run_map(Xz[idx], y[idx], Xtestz, ytest, lam=10)
print(f"\nReduced training set ({frac*100:.0f}%): Train={acc_small[0]:.3f}, Test={acc_small[1]:.3f}")

# Cross-validation stability check
print("\nCross-validation on full dataset:")
cv_train, cv_val, cv_norm = cross_validate_map(Xz, y, lam=10, k=5)


#  PyTorch Logistic Regression + Optimizer comparison
def make_loader(X, y, batch_size, shuffle=True):
    ds = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class LogisticRegression(nn.Module):
    def __init__(self, D, C, with_bias=False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(D, C) / math.sqrt(D))
        self.b = nn.Parameter(torch.zeros(C)) if with_bias else None
        self.with_bias = with_bias
    def forward(self, x):
        logits = x @ self.W + (self.b if self.with_bias else 0)
        return nn.functional.log_softmax(logits, dim=1)

def train_torch(opt_name="Adam", lr=1e-2, batch=64, epochs=50, wd=0.0, mom=0.9):
    model = LogisticRegression(Xz.shape[1], 2)
    crit = nn.NLLLoss(reduction="sum")
    optimizer = {
        "SGD": lambda: torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd),
        "Adam": lambda: torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd),
        "RMSprop": lambda: torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd),
        "AdamW": lambda: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd),
    }[opt_name]()
    loader = make_loader(Xz, y, batch)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optimizer.step()
    w = (model.W[:,1] - model.W[:,0]).detach().numpy()
    acc_train = np.mean(classify(Xz, w) == y)
    acc_test  = np.mean(classify(Xtestz, w) == ytest)
    return acc_train, acc_test, np.linalg.norm(w)

optimizers = ["SGD", "Adam", "RMSprop", "AdamW"]
results = []
for opt in optimizers:
    for lr in [1e-3, 1e-2, 5e-2]:
        for wd in [0.0, 1e-4, 1e-3]:
            atr, ats, wn = train_torch(opt, lr=lr, wd=wd)
            results.append({"opt": opt, "lr": lr, "wd": wd, "train_acc": atr, "test_acc": ats, "w_norm": wn})

df = pd.DataFrame(results).sort_values("test_acc", ascending=False)
print("\nTop configurations:\n", df.head())
#%%
# visualize
best_per_opt = df.groupby("opt", as_index=False).head(1)

plt.figure(figsize=(6,5))
bars = plt.bar(best_per_opt["opt"], best_per_opt["test_acc"])
plt.ylabel("Best Test Accuracy", fontsize=20)
plt.title("Optimizer Comparison", fontsize=20)
plt.ylim(0, 1)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        height, 
        f"{height:.3f}",   
        ha='center', va='bottom', fontsize=18
    )
plt.tick_params(axis='both', which='major', labelsize=16)

plt.show()

