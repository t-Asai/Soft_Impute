import numpy as np
import random
from methods_matrix import to_low_rank_matrix


def split_to_test_and_train(X, rate):
    """
    再構成がうまく出来ているかどうかを調べるために、訓練データとテストデータに分ける
    0の値は0を観測したのか観測できなかったのかの判断が面倒なので、
    この実装では0は観測できなかったとみなす。
    よって、比較するためには非ゼロの値を分離して比較する必要がある。
    """
    List = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] != 0:
                List.append([i, j])

    X_train = np.zeros(X.shape)
    X_test = np.zeros(X.shape)
    random.shuffle(List)
    for e, elem in enumerate(List):
        i = elem[0]
        j = elem[1]
        if e < len(List) * rate:
            X_train[i, j] = X[i, j]
        else:
            X_test[i, j] = X[i, j]
    return X_train, X_test


def make_sampling_matrix(X, rate):
    """
    観測行列を作成する関数
    テスト用にとっておいた部分は、観測できなかったことにする
    rateの意味は、本来観測できたであろう部分から、もっと差っ引きたかったら1より小さくすればいい
    """
    R = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if random.random() < rate and X[i, j] == 0:
                R[i, j] = 1.0
    return R


def make_target_matrix(N, rho):
    """
    アルゴリズムが正しいかどうかを調べるために、低ランクな行列を作成する関数
    """
    X0 = np.random.random((N, N))
    X0 = to_low_rank_matrix(X0, rho)
    return X0
