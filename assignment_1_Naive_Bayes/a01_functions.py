





# %%
import numpy as np
def logsumexp(x):
    """Computes log(sum(exp(x)).

    Uses offset trick to reduce risk of numeric over- or underflow. When x is a
    1D ndarray, computes logsumexp of its entries. When x is a 2D ndarray,
    computes logsumexp of each column.

    Keyword arguments:
    x : a 1D or 2D ndarray
    """
    offset = np.max(x, axis=0)
    return offset + np.log(np.sum(np.exp(x - offset), axis=0))


# %%
def nb_train(X, y, alpha=1, K=None, C=None):
    """Train a Naive Bayes model.

    We assume that all features are encoded as integers and have the same domain
    (set of possible values) from 0:(K-1). Similarly, class labels have domain
    0:(C-1).

    Parameters
    ----------
    X : ndarray of shape (N,D)
        Design matrix.
    y : ndarray of shape (N,)
        Class labels.
    alpha : int
        Parameter for symmetric Dirichlet prior (Laplace smoothing) for all
        fitted distributions.
    K : int
        Each feature takes values in [0,K-1]. None means auto-detect.
    C : int
        Each class label takes values in [0,C-1]. None means auto-detect.

    Returns
    -------
    A dictionary with the following keys and values:

    logpriors : ndarray of shape (C,)
        Log prior probabilities of each class such that logpriors[c] contains
        the log prior probability of class c.

    logcls : ndarray of shape(C,D,K)
        A class-by-feature-by-value array of class-conditional log-likelihoods
        such that logcls[c,j,v] contains the conditional log-likelihood of value
        v in feature j given class c.
    """
    X = np.asarray(X, dtype=int)
    y = np.asarray(y, dtype=int).ravel()
    N, D = X.shape
    if K is None:
        K = np.max(X) + 1
    if C is None:
        C = np.max(y) + 1

    # Compute class priors and store them in priors
    priors = np.zeros(C, dtype=float)
    # YOUR CODE HERE
    
    for c in range(C):
        Nc = np.sum(y == c)
        priors[c] = (Nc + (alpha - 1)) / (N + C * (alpha - 1))

    # Compute class-conditional densities in a class x feature x value array
    # and store them in cls.
    cls = np.zeros((C, D, K), dtype=float)
    # YOUR CODE HERE
    for c in range(C):
        Xc = X[y == c]
        Nc = Xc.shape[0]
        if Nc == 0:
            continue
        for j in range(D):
            counts = np.bincount(Xc[:, j], minlength=K)
            cls[c, j, :] = (counts + (alpha - 1)) / (Nc + K * (alpha - 1))

    # Apply epsilon only when smoothing is used
    if alpha > 1:
        priors[priors == 0] = np.finfo(float).tiny
        cls[cls == 0] = np.finfo(float).tiny

    # Output result
    return dict(logcls=np.log(cls), logpriors=np.log(priors))


# %%
def nb_predict(model, Xnew):
    """Predict using a Naive Bayes model.

    Parameters
    ----------
    model : dict
        A Naive Bayes model trained with nb_train.
    Xnew : nd_array of shape (Nnew,D)
        New data to predict.

    Returns
    -------
    A dictionary with the following keys and values:

    yhat : nd_array of shape (Nnew,)
        Predicted label for each new data point.

    logprob : nd_array of shape (Nnew,)
        Log-probability of the label predicted for each new data point.
    """
    logpriors = model["logpriors"]
    logcls = model["logcls"]
    Nnew = Xnew.shape[0]
    C, D, K = logcls.shape

    # Compute the unnormalized log joint probabilities P(Y=c, x_i) of each
    # test point (row i) and each class (column c); store in logjoint
    logjoint = np.zeros((Nnew, C))
    yhat = np.zeros(Nnew).astype(np.int64)
    logprob = np.zeros(Nnew)
    # YOUR CODE HERE
    for i in range(Nnew):
        for c in range(C):
            logjoint[i, c] = logpriors[c] + np.sum(logcls[c, np.arange(D), Xnew[i, :]])

    # Compute predicted labels (in "yhat") and their log probabilities
    # P(yhat_i | x_i) (in "logprob")
    # YOUR CODE HERE
    for i in range(Nnew):
        yhat[i] = np.argmax(logjoint[i, :])
        offset = np.max(logjoint[i, :])
        logprob[i] = logjoint[i, yhat[i]] - (offset + np.log(np.sum(np.exp(logjoint[i, :] - offset))))

    return dict(logprob=logprob, yhat=yhat)


# %%
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
    

