import numpy as np
import sys
from methods_graph import add_val, plot_val


def cal_total_error(X_k='', X_Original='', flag=''):
    """
    全体の誤差を評価する
    """
    func_name = sys._getframe().f_code.co_name
    if flag == '':
        num = np.linalg.norm(X_k - X_Original)
        den = np.linalg.norm(X_Original)
        val = num / den
        add_val(func_name, val)
        return val

    elif flag == 'plot':
        plot_val(func_name)

    elif flag == 'init':
        add_val(func_name, flag=flag)


def cal_test_error(X_k='', X_test='', flag=''):
    """
    テスト用に分離させて置いた部分のみの誤差を評価する
    """
    func_name = sys._getframe().f_code.co_name
    if flag == '':
        num = 0.0
        N, M = X_test.shape
        for i in range(N):
            for j in range(M):
                if X_test[i, j] != 0:
                    num += pow(X_test[i, j] - X_k[i, j], 2.0)
        den = np.linalg.norm(X_test)
        val = np.sqrt(num) / den
        add_val(func_name, val)
        return val

    elif flag == 'plot':
        plot_val(func_name)

    elif flag == 'init':
        add_val(func_name, flag=flag)


def cal_terminal_condition(X_k='', X_p='', flag=''):
    """
    アルゴリズムの終端条件を満たすかどうか評価する
    """
    func_name = sys._getframe().f_code.co_name
    if flag == '':
        num = np.linalg.norm(X_k - X_p)
        den = np.linalg.norm(X_p)
        val = num / den if den > 0 else 0.0
        add_val(func_name, val)
        return val

    elif flag == 'plot':
        plot_val(func_name)

    elif flag == 'init':
        add_val(func_name, flag=flag)
