# -*- coding: utf-8 -*-
import numpy as np
import random
from matrix_class import Matrix
import random


def soft_threshold(s, Lambda):
    if abs(s) <= Lambda:
        return 0
    elif s > 0:
        return s - Lambda
    else:
        return s + Lambda


def soft_impute(Y, X_train):
    Lambda = pow(10, 2)
    X_k = np.zeros(X_train.shape)
    while(Lambda > 0.01):
        error = 0.0
        _error = 0.0
        while(1):
            U, S, V = np.linalg.svd(X_k, full_matrices=True)
            S_ = np.array([soft_threshold(s, Lambda) for s in S])
            # print(S, S2)
            X_k = np.dot(np.dot(U, np.diag(S_)), V)
            for i in range(Y.shape[0]):
                for j in range(Y.shape[1]):
                    if(Y[i, j] != 0):
                        X_k[i, j] = Y[i, j]
            error = np.linalg.norm(X_k - X_train) / np.linalg.norm(X_train)
            print(error, end='\r')
            if abs(error - _error) < 0.1:
                Lambda *= 0.9
                break
            else:
                _error = error
        print(Lambda, error)


def split_to_test_and_train(X, rate):
    List = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] != 0:
                List.append([i, j])

    X_train = np.zeros(X.shape)
    X_test = np.zeros(X.shape)
    random.shuffle(List)
    for l in range(len(List)):
        elem = List[l]
        i = elem[0]
        j = elem[1]
        if l < len(List) * rate:
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


if __name__ == '__main__':
    """
    初期値を作る
    """
    N = 500
    X0 = np.random.random((N, N))
    U, S, V = np.linalg.svd(X0, full_matrices=True)
    S_ = np.array([s if i < 0.1 * N else 0 for i, s in enumerate(S)])
    X0 = np.dot(np.dot(U, np.diag(S_)), V)

    X_train, X_test = split_to_test_and_train(X0, 1.0)
    R = make_sampling_matrix(X_test, 0.8)

    Y = R * X_train
    soft_impute(Y, X_train)
