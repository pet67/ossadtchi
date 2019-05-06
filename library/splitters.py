import numpy as np


def train50_test50(X, Y):
    middle_index = int(0.5 * X.shape[0])
    X_train = np.copy(X[:middle_index])
    Y_train = np.copy(Y[:middle_index])
    X_test = np.copy(X[middle_index:])
    Y_test = np.copy(Y[middle_index:])
    yield X_train, Y_train, X_test, Y_test
