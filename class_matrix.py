import numpy as np
import random


class Matrix:

    def __init__(self, N=100, M=100, rate_dense=0.1, rate_sample=0.7, rate_train_test=0.3, noise_rate=0.0):
        self.N = N
        self.M = M
        self.rate_dense = rate_dense
        self.rate_sample = rate_sample
        self.rate_train_test = rate_train_test
        self.noise_rate = noise_rate

    def make_target_matrix(self):
        """
        アルゴリズムが正しいかどうかを調べるために、低ランクな行列を作成する関数
        """
        X = np.random.random((self.N, self.M))
        U, S, V = np.linalg.svd(X, full_matrices=False)
        S_ = np.array([s if i < self.rate_dense *
                       self.N else 0 for i, s in enumerate(S)])
        X = np.dot(np.dot(U, np.diag(S_)), V)
        self.Original = X

    def make_train_and_test_matrix(self):
        """
        再構成がうまく出来ているかどうかを調べるために、訓練データとテストデータに分ける
        0の値は0を観測したのか観測できなかったのかの判断が面倒なので、
        この実装では0は観測できなかったとみなす。
        よって、比較するためには非ゼロの値を分離して比較する必要がある。
        """
        List = []
        for i in range(self.N):
            for j in range(self.M):
                if self.Original[i, j] != 0:
                    List.append([i, j])
        random.shuffle(List)

        num_train = len(List) * self.rate_train_test
        self.Train = np.zeros((self.N, self.M))
        self.Test = np.zeros((self.N, self.M))
        for e, (i, j) in enumerate(List):
            if e < num_train:
                self.Train[i, j] = self.Original[i, j]
            else:
                self.Test[i, j] = self.Original[i, j]

    def make_sampling_matrix(self):
        """
        観測行列を作成する関数
        テスト用にとっておいた部分は、観測できなかったことにする
        rateの意味は、本来観測できたであろう部分から、もっと差っ引きたかったら1より小さくすればいい
        """
        self.Cast = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in range(self.M):
                if self.Test[i, j] == 0 and random.random() < self.rate_sample:
                    self.Cast[i, j] = 1.0

    def make_observed_matrix(self):
        self.Observe = self.Cast * self.Train
        noise_matrix = np.random.normal(0, 1, (self.N, self.M))
        self.Observe += self.noise_rate * noise_matrix
