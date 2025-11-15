
import numpy as np

def nb_generate(model, ygen):
    """Given a Naive Bayes model, generate some data.

    Parameters
    ----------
    model : dict
        A Naive Bayes model trained with nb_train.
    ygen : nd_array of shape (n,)
        Vector of class labels for which to generate data.

    Returns
    -------
    nd_array of shape (n,D)

    Generated data. The i-th row is a sampled data point for the i-th label in
    ygen.
    """
    logcls = model["logcls"]
    n = len(ygen)
    C, D, K = logcls.shape
    Xgen = np.zeros((n, D), dtype=int)
    for i in range(n):
        c = ygen[i]
        # Generate the i-th example of class c, i.e., row Xgen[i,:]. To sample
        # from a categorical distribution with parameter theta (a probability
        # vector), you can use np.random.choice(range(K),p=theta).
        # YOUR CODE HERE
        # Compute normalization manually (same logic as your logsumexp)
        logp = logcls[c]                      # shape (D, K)
        offset = np.max(logp, axis=1, keepdims=True)
        probs = np.exp(logp - offset)         # avoid overflow
        probs /= probs.sum(axis=1, keepdims=True)  # normalize per feature

        # Sample one pixel value per feature
        for j in range(D):
            Xgen[i, j] = np.random.choice(np.arange(K), p=probs[j])

    return Xgen