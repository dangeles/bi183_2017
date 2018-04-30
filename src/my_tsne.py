import numpy as np

def shannon(data, sigma=1.0):
    """Given data (squared differences of vectors), return the entropy and p_ij values for the data."""
    # Compute P-row and corresponding perplexity
    arg = -data/(2*sigma**2)

    if (arg > 0).any():
        raise ValueError('At least one probability is negative')

    if (arg > 710).any():
        raise ValueError('overflow warning, sigma={0:.2g}'.format(sigma))

    P = np.exp(arg)
    sumP = P.sum(axis=0)

    # H = -Sum_j p_jilogp_ji
    # p_ji = P/sumP
    # log p_ji = log P - log sumP
    # H = Sum_j p_ji/sumP * (D_ji/2*sigma**2 + np.log(sumP))
    # H = Sum_j (p_ji*D_ji/2*sigma**2))/sumP + p_ji/sumP*np.log(sumP)
    # H = beta * Sum_j (p_ji*D_ji)/sumP + Sum_j p_ji/sumP *np.log(sumP)
    # Sum_j p_ji = Sum_j p(j|i) = 1
    # H = beta * meancondD + np.log(sumP)
    H = np.log(sumP) + (2*sigma**2) * np.sum(data * P) / sumP

    if np.abs(H) == np.inf:
        raise ValueError('Entropy is undefined')

    # normalize the p_ij
    P = P/sumP
    return H, P

def binary_search(D_i, target, inv_sigma=1., inv_sigma_min=1.*10**-8,
                  inv_sigma_max=np.inf, tol=10**-3, max_iters=100):
    """Implement a binary search to find the ideal sigma_i."""
    H, P_i = shannon(D_i, 1/inv_sigma)
    # Evaluate whether the perplexity is within tolerance
    delta = H - target
    iterations = 0
    prevH = 0

    if type(tol) is not float:
        raise ValueError('tolerance value must be a number')

    while np.abs(delta) > tol:
        if iterations > max_iters:
            break
        if delta > 0:
            # if difference is positive, the minimum bound of sigma
            # is the current sigma:
            inv_sigma_min = inv_sigma
            # if sigmamax is at a boundary point:
            if inv_sigma_max == np.inf:
                # increase the current sigma to twice its value
                # (sigmamax is too high to average)
                inv_sigma = inv_sigma_min * 2.
            else:
                # otherwise take the average of bounds
                inv_sigma = (inv_sigma_min + inv_sigma_max)/2.
        else:
            inv_sigma_max = inv_sigma
            inv_sigma = (inv_sigma_min + inv_sigma_max)/2.

        # Update
        H, P_i = shannon(D_i, 1/inv_sigma)
        delta = H - target
        iterations += 1

        if prevH == H:
            return P_i, 1/inv_sigma
        prevH = H

        if iterations == 50:
            print('Error, non convergence')

    return P_i, 1/inv_sigma


def sne(X):
    """
    # calculate the dotproduct between each sample:
    # calculate |x_j|^2 for each vector
    """
    sum_X = np.sum(np.square(X), 1)
    dotprod = -2 * np.dot(X, X.T)
    # calculate
    # |x_j|^2 - 2*|x_i||x_j|cosTheta = |x_j|^2 - 2*|x_i - x_j|^2
    # this is asymmetric
    Dprime = np.add(dotprod, sum_X)
    # symmetrize by completing the square:
    # |x_j|^2 - 2*|x_i - x_j|^2 + |x_i|
    D = np.add(Dprime.T, sum_X)

    # set D_ii = 0
    D = D.astype(np.float)
    D = np.maximum(D, 0)
    np.fill_diagonal(D, 0)
    return D


def tsne_Y(Y):
    """
    # The code below changes between t-SNE and SNE.

    # The matrix below results in (p_ij + p_ji)/2 after exp and normalization
    # calculate the dotproduct between each sample:
    # calculate |x_j|^2 for each vector
    """
    sum_Y = np.sum(np.square(Y), 1)
    dotprod = -2 * np.dot(Y, Y.T)
    # calculate
    # |x_j|^2 - 2*|x_i||x_j|cosTheta = |x_j|^2 - 2*|x_i - x_j|^2
    # this is asymmetric
    Dprime = np.add(dotprod, sum_Y)
    # symmetrize by completing the square:
    # |x_j|^2 - 2*|x_i - x_j|^2 + |x_i|
    D = np.add(Dprime.T, sum_Y)

    # student t with 1df
    numerator = 1/(1 + D)
    Q = numerator/numerator.sum(axis=0)

    # underflow
    Q = np.maximum(Q, 10**-12)
    np.fill_diagonal(Q, 0)
    np.fill_diagonal(numerator, 0)
    return Q, numerator






def run_SNE(X=np.array([]), no_dims=2, perplexity=30.0, reduce_dims=0, max_iter=1000, learning_rate=1., SNE=True, min_gain=0.1):
    """Run t-sne on dataset."""
    # if desired, PCA reduce the data
    if reduce_dims != 0:
        X, _, _ = pca(X, reduce_dims)
        print(X.max())
        print(X.sum(axis=1).max())

    # initialize variables
    n, d = X.shape
    min_gain = 0.01  # minimum gain
    initial_momentum = 0.5
    final_momentum = 0.8

    # initialize Y matrix:
    Y = np.random.randn(n, no_dims)  # Y shaped as samples(n) and no_dims (50)
    # initialize gradient wrt Y matrix
    gradY = np.zeros((n, no_dims))  # deltaY
    diffY = np.zeros((n, no_dims))  # for gradient computations
    iY = np.zeros((n, no_dims))  # for gradient computations
    gains = np.ones((n, no_dims))  # no clue
    KL = np.zeros(max_iter)

    # Compute P-values
    P = find_perplexity(X, perplexity=perplexity, tol=1.*10**-3)
    if SNE == False:
        # symmetrize by adding p_ij + p_ji
        P = P + np.transpose(P)
        # normalization will take care of the
        # extra factor of 2
        P = P/P.sum(axis=1)

    # make sure off-diagonals are not zero
    # underflow is a real problem here
    P = np.maximum(P, 10**-20)
    np.fill_diagonal(P, 0)
    p = 4*P
    # Run iterations
    for k in range(max_iter):

        # Compute pairwise affinities
        if SNE == True:
            Dy = sne(Y)
            # normalize
            Q = Dy/Dy.sum(axis=0)
        else:
            Q, numerator = tsne_Y(Y)

        # Compute gradient
        if SNE == True:
            deltaPQ = P + P.T - Q - Q.T
        else:
            deltaPQ = P - Q

        for i in range(n):
            if SNE == True:
                # calculate Y_i - Y_j for all j
                # This is the same as adding Y_i to the matrix of -Y
                difference = (Y[i, :] - Y)
                # we are currently calculating gradY for the y_i
                # P_i = {p(i|j)} \in j
                # but we need p(j|i), so select P.T_i instead
                gradY[i, :] = np.dot(deltaPQ[:, i], difference)
            else:
                dPQ = deltaPQ[:, i]*numerator[:, i]
                gradY[i, :] = dPQ.dot(Y[i, :] - Y)

        # Perform the update
        diffY += learning_rate*gradY

        # Renormalize, otherwise everything explodes. >.<
        # I spent a really long time figuring this out.
        # This feels like something the exam should have
        # told us.
        if k < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((gradY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((gradY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain

        iY = momentum * iY -  (gains * diffY)
#         Y = Y + diffY*gains

        # add jitter:
        Y += gains
#         Y += np.random.normal(0, 10**-5, Y.shape)
        Y = (Y-Y.mean(axis=0)[np.newaxis, :])/np.abs(Y).max(axis=1).reshape(n, 1)

        if np.isnan(Y).any():
            raise ValueError('Y has become undefined')
        if (np.abs(Y) > 10**4).any():
            raise ValueError('Y is exploding')

        # Kullback Leibler
        padP = P.copy()
        np.fill_diagonal(padP, 10**-12)
        padQ = Q.copy()
        np.fill_diagonal(padQ, 10**-12)
        KL[k] = np.sum(padP * np.log(padP / padQ))

        if k%100 == 0:
            print('Iteration {0}/{1}'.format(k, max_iter))
            print(np.max(deltaPQ))

        if k > 50:
            P /= 4

    # Return solution
    return Y, P, Q, KL
