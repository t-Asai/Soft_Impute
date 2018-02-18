import numpy as np
from methods_cal_param import cal_total_error, cal_test_error, cal_terminal_condition


def warm_start(Mat, Lambda_param, stop_condition):
    """
    Soft_Imputeをcold startさせないための方法
    """

    """
    何度もclassを呼び出す必要が無いと思ったので、分離して見た
    """
    X_Original = Mat.Original
    X_train = Mat.Train
    X_test = Mat.Test
    Y = Mat.Observe
    R = Mat.Cast

    """
    グラフ描画用のデータの初期化
    """
    cal_total_error(flag='init')
    cal_test_error(flag='init')
    cal_terminal_condition(flag='init')

    X_k = np.zeros(X_train.shape)
    Lambda = Lambda_param.start

    while(Lambda > Lambda_param.end):
        X_k = soft_impute(Y, R, X_k, Lambda, stop_condition)
        total_error = cal_total_error(X_k, X_Original)
        test_error = cal_test_error(X_k, X_test)
        print('Lambda: {:.3g}, total_error: {:.3g}, test_error: {:.3g}'.format(
            Lambda, total_error, test_error))
        Lambda *= Lambda_param.ratio

    """
    グラフの描画
    """
    cal_total_error(flag='plot')
    cal_test_error(flag='plot')

    return X_k


def soft_impute(Y, R, X_k, Lambda, stop_condition):
    """
    アルゴリズムのメイン
    """

    val = stop_condition
    N, M = Y.shape

    while(val >= stop_condition):
        X_p = X_k

        U, S, V = np.linalg.svd(X_k, full_matrices=False)
        S_ = np.array([soft_threshold(s, Lambda) for s in S])
        X_k = np.dot(np.dot(U, np.diag(S_)), V)
        """
        ここの書き方がきたないから嫌いだが、速く綺麗な書き方が分からない
        """
        for i in range(N):
            for j in range(M):
                if(R[i, j] != 0):
                    X_k[i, j] = Y[i, j]
        """
        X_k = np.array([Y[i, j] if R[i, j] != 0 else X_k[i, j]
                        for i in range(Y.shape[0]) for j in range(Y.shape[1])]).reshape(Y.shape)
        """
        val = cal_terminal_condition(X_k, X_p)
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
