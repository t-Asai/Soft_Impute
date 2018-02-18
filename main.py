# import pandas as pd
from collections import namedtuple
from methods_algorithm import warm_start
from class_matrix import Matrix
"""
メインの関数
"""

if __name__ == "__main__":

    """
    パラメータの設定
    """
    Mat = Matrix(N=100, M=100, rate_dense=0.1,
                 rate_sample=0.9, rate_train_test=0.9, noise_rate=0.0)

    LAMBDA = namedtuple('LAMBDA', ('start', 'end', 'ratio'))
    Lambda_param = LAMBDA(start=pow(10, 2), end=pow(10, -5), ratio=0.9)

    stop_condition = pow(10, -1)

    """
    初期値作成
    外部からデータを入れる場合は、そのデータのサイズに合わせてパラメータを設定して
    Mat.Originalを上書きすればいけるはず
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
    X_k = warm_start(Mat, Lambda_param, stop_condition)
