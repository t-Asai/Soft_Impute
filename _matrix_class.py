# -*- coding: utf-8 -*-
import numpy as np
import random


class Matrix:
    col = 100
    row = 100
    rank = 3
    sparsity = 0.8

    def make_low_rank_matrix(self):
        A = np.random.normal(0, 1, (self.col, self.row))
        U, S, V = np.linalg.svd(A, full_matrices=True)
        S2 = np.array([s if i < self.rank else 0.0 for i, s in enumerate(S)])
        return np.dot(np.dot(U, np.diag(S2)), V)

    def lack_data(self, A):
        A2 = np.empty(A.shape)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A2[i, j] = 0.0 if random.random() < self.sparsity else A[i, j]
        return A2

    def cal_error(self, A1, A2):
        print(np.linalg.norm(A1 - A2))
