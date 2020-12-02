import numpy as np
import matplotlib.pyplot as plt

coefs, X = chebyshev(np.exp, 3, -10, 10, 500)

yhat = X @ coefs
ytrue = np.exp(X[:, 1])
plt.plot(X[:, 1], ytrue)
plt.plot(X[:, 1], yhat)
plt.legend(["true", "predicted"])
