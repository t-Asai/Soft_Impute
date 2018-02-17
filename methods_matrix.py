import numpy as np


def to_square_matrix(X):
    if X.shape[0] > X.shape[1]:
        return np.sqrt(np.dot(X, X.T))
    elif X.shape[0] < X.shape[1]:
        return np.sqrt(np.dot(X.T, X))
    else:
        return X


def to_low_rank_matrix(X, rho):
    N = X.shape[0]
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S_ = np.array([s if i < rho * N else 0 for i, s in enumerate(S)])
    print(U.shape, S.shape, V.shape)
    X = np.dot(np.dot(U, np.diag(S_)), V)
    return X
