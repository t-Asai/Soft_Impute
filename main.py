import make_matrix
from methods import warm_start

if __name__ == "__main__":
    """
    メインの関数
    """
    N = 500
    s_Lambda = pow(10, 2)
    r_Lambda = 0.9
    e_Lambda = 0.01
    rho = 0.1
    under_sampling_rate = 0.8
    test_train_ratio = 1.0
    stop_condition = 0.1

    X0 = make_matrix.make_target_matrix(N, rho)
    X_train, X_test = make_matrix.split_to_test_and_train(X0, test_train_ratio)
    R = make_matrix.make_sampling_matrix(X_test, under_sampling_rate)

    Y = R * X_train
    warm_start(Y, X_train, s_Lambda, r_Lambda, e_Lambda, stop_condition)
