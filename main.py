import make_matrix
from methods import warm_start

if __name__ == "__main__":
    N = 500
    Lambda = pow(10, 2)

    X0 = make_matrix.make_target_matrix(N)
    X_train, X_test = make_matrix.split_to_test_and_train(X0, 1.0)
    R = make_matrix.make_sampling_matrix(X_test, 0.8)

    Y = R * X_train
    warm_start(Y, X_train, Lambda)
