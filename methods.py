import numpy as np


def warm_start(Y, X_train, X_test, s_Lambda, r_Lambda, e_Lambda, stop_condition):
    """
    Soft_Imputeをcold startさせないための方法
    """
    X_k = np.zeros(X_train.shape)
    Lambda = s_Lambda
    while(Lambda > e_Lambda):
        X_k = soft_impute(
            Y, X_k, X_train, Lambda, stop_condition)
        total_error = cal_total_error(X_k, X_train, X_test)
        error = cal_test_error(X_k, X_test)
        print('Lambda: {:.3g}, total_error: {:.3g}, test_error: {:.3g}'.format(
            Lambda, total_error, error))
        Lambda *= r_Lambda
    return X_k


def soft_impute(Y, X_k, X_train, Lambda, stop_condition):
    """
    アルゴリズムのメイン
    """
    while(1):
        X_p = X_k
        U, S, V = np.linalg.svd(X_k, full_matrices=True)
        S_ = np.array([soft_threshold(s, Lambda) for s in S])
        # print(S, S2)
        X_k = np.dot(np.dot(U, np.diag(S_)), V)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if(Y[i, j] != 0):
                    X_k[i, j] = Y[i, j]

        if cal_terminal_condition(X_k, X_p) < stop_condition:
            break
    return X_k


def soft_threshold(s, Lambda):
    """
    弱閾値関数
    """
    if abs(s) <= Lambda:
        return 0.0
    elif s > 0:
        return s - Lambda
    else:
        return s + Lambda


def cal_total_error(X_k, X_train, X_test):
    error = np.linalg.norm(X_k - X_train - X_test) / np.linalg.norm(X_train)
    return error


def cal_test_error(X_k, X_test):
    error = 0.0
    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            if X_test[i, j] != 0:
                error += pow(X_test[i, j] - X_k[i, j], 2.0)
    error /= np.linalg.norm(X_test)
    return error


def cal_terminal_condition(X_k, X_p):
    num = np.linalg.norm(X_k - X_p)
    den = np.linalg.norm(X_p)
    return num / den if den > 0 else 0.0
