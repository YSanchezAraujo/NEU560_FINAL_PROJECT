def composeMaps(x, origA, origB, destA, destB):
    """
    inputs:
        x: points in the original space
        origA: minimum of the space of x
        origB: maximum of the space of x
        destA: minimum of the space of y
        destB: maximum of the space of y
    returns:
        points: y mapped from x in [origA, origB] to y in [destA, destB]
    """
    x = np.asarray(x)
    maxX, minX = np.max(x), np.min(x)
    assert maxX <= origB
    assert minX >= origA
    return destA + ((destB - destA) / (origB - origA)) * (x - origA)

def chebyshev(g, m, dMin, dMax, nPoints):
    """
    inputs:
       g: the function to appr    T = np.ones((len(data), m+1))oximate
       m: degree of polynomial
       dMax: max value of data ~ g()
       dMin: min value of data ~ g()
       nPoints: number of points to use in approximation
    returns:
       c: chebyshev coefficients to approximate g()
    """
    data = np.linspace(dMin, dMax, nPoints)
    data_on_P = composeMaps(data, dMin, dMax, -1, 1)
    T = np.ones((len(data), m+1))
    T[:, 1] = np.copy(data) # now it has the T_0 and T_1 terms
    for k in range(1, m):
        # recursively add the rest
        T[:, k+1] = 2 * data * T[:, k] - T[:, k-1]
    W = np.diag(w(data_on_P))
    S = T.T @ W @ T
    u = T.T @ W @ g(data)
    c = np.linalg.lstsq(S, u, rcond=None)
    return c[0]

def w(x):
    """
    inputs:
        x: set of ponts
    returns:
        y: points in [-1, 1] weighted by a function 1/sqrt(1 - x^2)
    """
    return 1 / (np.sqrt(1 - x**2) + np.finfo(float).eps)


def mle_poisson_approx(X, y, c_s, M_nc):
    """
    doc: todo
    """
    M = c_s[-1] * M_nc
    M_inv = np.linalg.inv(M)
    return 0.5 * M_inv @ X.T @ (y - c_s[1])

def negLogLikPoisson(w, X, y):
    f = np.exp(X @ w)
    return -(y.T @ np.log(f) - np.sum(f))

def fit_glm(X, y, dMin, dMax, nPoints, adapt, nAdapt):
    """
    doc: todo
    """
    m = 2 # 2 is order of poly
    M_nc = X.T @ X # save time on this because it only needs to be computed once
    if adapt:
        dMins = np.linspace(dMin-3, dMin+3, nAdapt)
        dMaxs = np.linspace(dMax-3, dMax+3, nAdapt)
        vals = list(zip(dMins, dMaxs))
        ll_grid = np.zeros(len(vals))
        W = np.zeros((len(vals), X.shape[1]))
        for k, j in enumerate(vals):
            c_s = chebyshev(np.exp, m, j[0], j[1], nPoints)
            w_est = mle_poisson_approx(X, y, c_s, M_nc)
            ll_grid[k] = -negLogLikPoisson(w_est, X, y)
            W[k, :] = w_est
        best_widx = np.argmax(ll_grid)
        return W[best_widx, :], ll_grid
    c_s = chebyshev(np.exp, m, dMin, dMax, nPoints)
    return mle_poisson_approx(X, y, c_s, M_nc)
