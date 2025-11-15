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
# # 5 Generating Data

# %%
# %load_ext autoreload
# %autoreload 2

from a01_helper import *
from a01_functions import nb_train, nb_generate
import warnings
warnings.filterwarnings('ignore')


# %%
# let's generate 15 digits from each class and plot

model_nb2 = nb_train(X, y, alpha=1.1) # alpha=1.1 from CV in task 4
ygen = np.repeat(np.arange(10), 15)
Xgen = nb_generate(model_nb2, ygen)

nextplot()
showdigits(Xgen, ygen)
plt.suptitle("Some generated digits for each class")

# %%
# we can also plot the parameter vectors by choosing the most-likely
# value for each feature
ymax = np.arange(10)
Xmax = np.zeros((10, D))
for c in range(10):
    Xmax[c,] = np.apply_along_axis(np.argmax, 1, model_nb2["logcls"][c, :, :])

nextplot()
showdigits(Xmax, ymax)
plt.suptitle("Most likely value of each feature per class")

# %%
# Or the expected value of each feature. Here we leave the categorical domain
# and treat each feature as a number, i.e., this is NOT how categorical Naive
# Bayes sees it and we wouldn't be able to do this if the data were really
# categorical.
ymean = np.arange(10)
Xmean = np.zeros((10, D))
for c in range(10):
    Xmean[c,] = np.apply_along_axis(
        np.sum, 1, np.exp(model_nb2["logcls"][c, :, :]) * np.arange(256)
    )

nextplot()
showdigits(Xmean, ymean)
plt.suptitle("Expected value of each feature per class")

# %%
# Compare generated digits for different α values 
alphas = [1, 2, 10, 20]
samples_per_class = 4
ygen = np.repeat(np.arange(10), samples_per_class)

fig, axes = plt.subplots(1, 4, figsize=(14, 6))
plt.suptitle("Effect of α on Generated Digits (4 Samples per Class)", fontsize=16, fontweight="bold")

for idx, alpha in enumerate(alphas):
    ax = axes[idx]
    model = nb_train(X, y, alpha=alpha)
    Xgen = nb_generate(model, ygen)

    # Plot digits directly using imshow
    grid_size = (10, samples_per_class)
    img_side = int(np.sqrt(Xgen.shape[1]))  # MNIST = 28

    canvas = np.zeros((10 * img_side, samples_per_class * img_side))
    for i in range(10):
        for j in range(samples_per_class):
            index = i * samples_per_class + j
            digit = Xgen[index, :].reshape(img_side, img_side)
            r0, r1 = i * img_side, (i + 1) * img_side
            c0, c1 = j * img_side, (j + 1) * img_side
            canvas[r0:r1, c0:c1] = digit

    ax.imshow(canvas, cmap="gray")
    ax.set_title(f"α = {alpha}")
    ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()


# %%
