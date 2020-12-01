import numpy as np
import scipy.integrate as integrate

"""
ALGORITHM 10.6: (least-squares approximation using orthoogal polynomials)
inputs: f(x) - a continuous founction f(x) on [a, b]
        w(x) - a weight function (integrable on [a, b])
        {psi_k(x)}_k=0^n - a set of n orthogonal functions on [a, b]

outpus: coeffcients a_0, a_1, ... , a_n such that the squared error is minimized

step 1. compute C_k, k = 0, 1, ... , n as:
    for k = 0, 1, ... , n do
        C_k = \int_a^b w(x) \psi_k^2(x) dx
    end

step 2. compute a_k, k = 0, 1, ... , n as:
    for k = 0, 1, ... , n do
        a_k = 1 / C_k \int_a^b w(x) f(x) \psi_k(x) dx
    end


Three-Term Recurrence Formula for Chebyshev Polynomials

    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x) , n >= 1
""";

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

def chebyshev(g, m, dMax, dMin, nPoints):
    """
    inputs:
       g: the function to approximate
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
    W = np.diagw(data_on_P)
    S =  T.T @ W @ T
    u = T.T @ W @ g(data)
    return np.linalg.lstsq(S, u)

def w(x):
    """
    inputs:
        x: set of ponts
    returns:
        y: points in [-1, 1] weighted by a function 1/sqrt(1 - x^2)
    """
    return 1 / np.sqrt(1 - x**2)

    
