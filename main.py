import make_matrix
from methods import warm_start, cal_test_error

if __name__ == "__main__":
    """
    メインの関数
    """
    N = 500
    s_Lambda = pow(10, 2)
    r_Lambda = 0.9
    e_Lambda = pow(10, -5)
    rho = 0.1
    under_sampling_rate = 1.0
    test_train_ratio = 0.7
    stop_condition = pow(10, -1)

    X0 = make_matrix.make_target_matrix(N, rho)
    X_train, X_test = make_matrix.split_to_test_and_train(X0, test_train_ratio)
    R = make_matrix.make_sampling_matrix(X_test, under_sampling_rate)

    Y = R * X_train
    X_k = warm_start(Y, X_train, X_test, s_Lambda,
                     r_Lambda, e_Lambda, stop_condition)
    cal_test_error(X_k, X_test)
