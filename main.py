import make_matrix
from methods_algorithm import warm_start
from methods_matrix import to_square_matrix, to_low_rank_matrix
import pandas as pd
import numpy as np
from collections import namedtuple
from plot_dat import plot_val


if __name__ == "__main__":
    """
    メインの関数
    長々と書いてるのをもっと切り分けたい
    """
    N = 100

    LAMBDA = namedtuple('LAMBDA', ('start', 'end', 'ratio'))
    Lambda_param = LAMBDA(start=pow(10, 2), end=pow(10, -5), ratio=0.9)
    rho = 0.1
    under_sampling_rate = 0.9
    test_train_ratio = 0.8

    stop_condition = pow(10, -1)

    X0 = make_matrix.make_target_matrix(N, rho)
    """
    X0 = pd.read_csv('data.csv')
    """

    # X0 = to_square_matrix(X0)
    X0 = to_low_rank_matrix(X0, rho)

    # 評価用にデータを分ける
    X_train, X_test = make_matrix.split_to_test_and_train(X0, test_train_ratio)
    # 再構成時に使用するデータをアンダーサンプリングする
    # 評価用に取り分けた部分は別枠で取り除くので、学習用のデータをさらに差っ引く場合にレートを下げる
    R = make_matrix.make_sampling_matrix(X_test, under_sampling_rate)

    # 作成した写像行列Rによって観測が得られる
    Y = R * X_train
    # Y += 0.01 * np.random.normal(0, 1, X0.shape)
    X_k = warm_start(Y, R, X_train, X_test, Lambda_param, stop_condition)

    plot_val('cal_total_error')
    plot_val('cal_test_error')
    plot_val('cal_terminal_condition')
