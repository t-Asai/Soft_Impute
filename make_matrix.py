import numpy as np
import random


def split_to_test_and_train(X, rate):
    List = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] != 0:
                List.append([i, j])

    X_train = np.zeros(X.shape)
    X_test = np.zeros(X.shape)
    random.shuffle(List)
    for e, elem in enumerate(List):
        i = elem[0]
        j = elem[1]
        if e < len(List) * rate:
            X_train[i, j] = X[i, j]
        else:
            X_test[i, j] = X[i, j]
    return X_train, X_test


def make_sampling_matrix(X, rate):
    R = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if random.random() < rate and X[i, j] == 0:
                R[i, j] = 1.0
    return R


def make_target_matrix(N):
    X0 = np.random.random((N, N))
    U, S, V = np.linalg.svd(X0, full_matrices=True)
    S_ = np.array([s if i < 0.1 * N else 0 for i, s in enumerate(S)])
    X0 = np.dot(np.dot(U, np.diag(S_)), V)
    return X0
