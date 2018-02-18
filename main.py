# import pandas as pd
from collections import namedtuple
import sys
import numpy as np
import math

from methods_algorithm import warm_start
from class_matrix import Matrix
from methods_graph import add_val, plot_vals
"""
メインの関数
"""

LAMBDA = namedtuple('LAMBDA', ('start', 'end', 'ratio'))


def SINGLE(Mat, Lambda_param):
    """
    パラメータの設定
    """

    stop_condition = pow(10, -1)

    """
    初期値作成
    外部からデータを入れる場合は、そのデータのサイズに合わせてパラメータを設定して
    Mat.Originalを上書きすればいけるはず
    交差検証用に関数を書き換えているので、
    実データでやりたい場合は、SINGLEに書いてあることを大体コピペして
    使いまわすと良いかな
    """
    Mat.make_target_matrix()
    # Mat.Original = pd.read_csv('data.csv')

    """
    評価用にデータを分離
    """
    Mat.make_train_and_test_matrix()

    """
    再構成時に使用するデータをアンダーサンプリングする
    評価用に取り分けた部分は別枠で取り除くので、学習用のデータをさらに差っ引く場合にレートを下げる
    """
    Mat.make_sampling_matrix()
    Mat.make_observed_matrix()

    """
    アルゴリズムを実行する
    """
    X_k, test_error, Lambda = warm_start(Mat, Lambda_param, stop_condition)
    return test_error, Lambda


def CROSS():
    func_name = sys._getframe().f_code.co_name
    add_val(func_name, flag='init')
    for x in range(-5, 2 + 1):
        cross_val = pow(10, x)
        errors = []
        for i in range(10):
            Mat = Matrix(N=100, M=100, rate_dense=0.1,
                         rate_sample=0.9, rate_train_test=0.9, noise_rate=0.001)

            Lambda_param = LAMBDA(start=cross_val * pow(10, 5),
                                  end=cross_val, ratio=0.9)
            test_error, Lambda = SINGLE(Mat, Lambda_param)
            print(Lambda, test_error)
            errors.append(test_error)
        add_val(func_name, val=[math.log(Lambda),
                                math.log(np.mean(errors))], flag='')
    plot_vals(func_name)


if __name__ == "__main__":
    """
    Mat = Matrix(N=100, M=100, rate_dense=0.1,
                 rate_sample=0.9, rate_train_test=0.9, noise_rate=0.0)

    Lambda_param = LAMBDA(start=pow(10, 2), end=pow(10, -5), ratio=0.9)
    SINGLE(Mat, Lambda_param)
    """

    CROSS()
