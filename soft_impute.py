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


def soft_impute():
    M = Matrix()
    A0 = M.make_low_rank_matrix()

    Lambda = pow(10, -1)
    A2 = M.lack_data(A0)
    A3 = A2
    while(1):
        U, S, V = np.linalg.svd(A3, full_matrices=True)
        S2 = np.array([soft_threshold(s, Lambda) for s in S])
        # print(S, S2)
        A3 = np.dot(np.dot(U, np.diag(S2)), V)
        for i in range(A0.shape[0]):
            for j in range(A0.shape[1]):
                if(A2[i, j] != 0):
                    A3[i, j] = A2[i, j]
        M.cal_error(A0, A3)


def split_to_test_and_train(X, rate):
    List = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] != 0:
                List.append([i, j])

    R_train = np.zeros(X.shape)
    R_test = np.zeros(X.shape)
    random.shuffle(List)
    for l in range(len(List)):
        elem = List[l]
        i = elem[0]
        j = elem[1]
        if l < len(List) * rate:
            R_train[i, j] = 1.0
        else:
            R_test[i, j] = 1.0
    return R_train, R_test


if __name__ == '__main__':
    A = np.random.random((10, 10))
    R_train, R_test = split_to_test_and_train(A, 0.1)
    print(R_train)
    print(R_test)
