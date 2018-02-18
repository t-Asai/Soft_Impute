import numpy as np


def cal_total_error(X_k, X_train, X_test):
    num = np.linalg.norm(X_k - X_train - X_test)
    den = np.linalg.norm(X_train)
    val = num / den
    return val


def cal_test_error(X_k, X_test):
    num = 0.0
    N, M = X_test.shape
    for i in range(N):
        for j in range(M):
            if X_test[i, j] != 0:
                num += pow(X_test[i, j] - X_k[i, j], 2.0)
    den = np.linalg.norm(X_test)
    val = np.sqrt(num) / den
    return val


def cal_terminal_condition(X_k, X_p):
    num = np.linalg.norm(X_k - X_p)
    den = np.linalg.norm(X_p)
    val = num / den if den > 0 else 0.0
    return val
