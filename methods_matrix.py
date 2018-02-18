import numpy as np


def to_square_matrix(X):
    """
    これはもう要らないかもしれない
    正則で無い行列をそのままSVDするよりは、
    転置かけてからSVDする方が良いかな？と思って作って見たけど
    特に面白いことにはならなかった
    そもそも、転置を掛けても固有値か特異値かぐらいだから
    影響はなくて当然か
    もしかしたら、強制的に小さい方に寄せることで対象の行列の観測密度？
    が高まるかもしれないので、一応置いておく
    """
    N, M = X.shape
    if N > M:
        return np.sqrt(np.dot(X, X.T))
    elif N < M:
        return np.sqrt(np.dot(X.T, X))
    else:
        return X


def to_low_rank_matrix(X, rho):
    N = X.shape[0]
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S_ = np.array([s if i < rho * N else 0 for i, s in enumerate(S)])
    X = np.dot(np.dot(U, np.diag(S_)), V)
    return X
