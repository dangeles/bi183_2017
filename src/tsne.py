import numpy as np


def Hbeta(D=np.array([]), sigma=1.0):
    """
    Compute the P_ji matrix and the entropy for some data given a sigma value.

    Params:
    D - Squared difference of two vectors. Must be a numpy array.
    sigma - a float

    Output:
    H, P - Entropy and P_ji matrix
    """
    # Compute P-row and corresponding perplexity
    P = np.exp(-D / (2*sigma**2))
    sumP = np.sum(P)
    # H = -Sum_j p_jilogp_ji
    # p_ji = P/sumP
    # log p_ji = log P - log sumP
    # H = Sum_j p_ji/sumP * (D_ji/2*sigma**2 + np.log(sumP))
    # H = Sum_j (p_ji*D_ji/2*sigma**2))/sumP + p_ji/sumP*np.log(sumP)
    # H = beta * Sum_j (p_ji*D_ji)/sumP + Sum_j p_ji/sumP *np.log(sumP)
    # Sum_j p_ji = Sum_j p(j|i) = 1
    # H = beta * meancondD + np.log(sumP)
    H = np.log(sumP) + (2*sigma**2) * np.sum(D * P) / sumP
    # normalize the perplexity
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """Binary search to optimize P_ji given a perplexity."""
    # Initialize some variables
    print("Computing pairwise distances...")
    # n = samples, d = dimensions
    (n, d) = X.shape
    # The code below changes between t-SNE and SNE
    # The matrix below results in (p_ij + p_ji)/2 after exp and normalization
    # calculate the dotproduct between each sample:
    # calculate |x_j|^2 for each vector
    sum_X = np.sum(np.square(X), 1)
    dotprod = -2 * np.dot(X, X.T)
    # calculate
    # |x_j|^2 - 2*|x_i||x_j|cosTheta = |x_j|^2 - 2*|x_i - x_j|^2
    # this is asymmetric
    Dprime = np.add(dotprod, sum_X)
    # symmetrize by completing the square:
    # |x_j|^2 - 2*|x_i - x_j|^2 + |x_i|
    D = np.add(Dprime.T, sum_X)
    # initialize a P matrix
    P = np.zeros((n, n))
    # initialize a sigma matrix
    sigma = np.ones((n, 1))
    # set the target value
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        sigmamin = 0
        sigmamax = np.inf

        # remake the data matrix, excluding the i'th column
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, sigma[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                # if difference is positive, the minimum bound of sigma
                # is the current sigma:
                sigmamin = sigma[i].copy()
                # if sigmamax is at a boundary point:
                if sigmamax == np.inf or sigmamax == 0:
                    # increase the current sigma to twice its value
                    # (sigmamax is too high)
                    sigma[i] = sigmamin * 2.
                else:
                    # otherwise take the average of bounds
                    sigma[i] = (sigmamin + sigmamax) / 2.
            else:
                sigmamax = sigma[i].copy()
                if sigmamin == np.inf or sigmamin == 0:
                    sigma[i] = sigmamax / 2.
                else:
                    sigma[i] = (sigmamin + sigmamax) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, sigma[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(sigma))
    return P


def pca(X=np.array([]), no_dims=50):
    """PCA on X."""
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    # mean center along columns
    X = X - np.tile(np.mean(X, 0), (n, 1))
    # get the eigenvalues of to covariance matrix
    # use eigh for hermitian/symmetric matrices ;)
    (l, M) = np.linalg.eigh(np.dot(X.T, X))
    # transform the coordinates
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
    Run t-sne on dataset.

        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500  # learning rate
    min_gain = 0.01  # minimum gain
    Y = np.random.randn(n, no_dims)  # Y shaped as samples(n) and no_dims (50)
    dY = np.zeros((n, no_dims))  # deltaY
    iY = np.zeros((n, no_dims))  # no clue
    gains = np.ones((n, no_dims))  # no clue

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)  # magnitude of Y along no_dims
        num = -2. * np.dot(Y, Y.T)  # 2*variance of Y
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y
