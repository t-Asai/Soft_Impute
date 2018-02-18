import pandas as pd
from collections import namedtuple
from methods_algorithm import warm_start
from methods_graph import plot_val
from class_matrix import Matrix

if __name__ == "__main__":
    """
    メインの関数
    長々と書いてるのをもっと切り分けたい
    """
    Mat = Matrix(N=100, M=100, rate_dense=0.1,
                 rate_sample=0.9, rate_train_test=0.9)

    LAMBDA = namedtuple('LAMBDA', ('start', 'end', 'ratio'))
    Lambda_param = LAMBDA(start=pow(10, 2), end=pow(10, -5), ratio=0.9)

    stop_condition = pow(10, -1)

    """
    初期値作成
    """
    Mat.make_target_matrix()
    Mat.split_to_test_and_train()
    Mat.make_sampling_matrix()

    """
    評価用にデータを分ける
    """
    X_train = Mat.Train
    X_test = Mat.Test
    Y = Mat.Observe
    R = Mat.Cast

    """
    再構成時に使用するデータをアンダーサンプリングする
    評価用に取り分けた部分は別枠で取り除くので、学習用のデータをさらに差っ引く場合にレートを下げる
    """

    """
    アルゴリズムを実行する
    """
    X_k = warm_start(Y, R, X_train, X_test, Lambda_param, stop_condition)

    """
    描画する
    """
    """
    plot_val('cal_total_error')
    plot_val('cal_test_error')
    plot_val('cal_terminal_condition')
    """
