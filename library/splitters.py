import numpy as np


def new_4x_cross_testing(X, Y):
    train_share = 0.6
    val_share = (1 - train_share) / 2
    test_share = (1 - train_share) / 2
    assert (train_share + val_share + test_share) == 1

    samples_in_data = int(X.shape[0])

    train_index = int(samples_in_data * train_share)
    val_index = int(samples_in_data * (train_share + val_share))
    direct_train_slice = slice(None, train_index)
    direct_val_slice = slice(train_index, val_index)
    direct_test_slice = slice(val_index, None)

    reversed_test_index = int(samples_in_data * test_share)
    reversed_val_index = int(samples_in_data * (test_share + val_share))
    reversed_train_slice = slice(reversed_val_index, None)
    reversed_val_slice = slice(reversed_test_index, reversed_val_index)
    reversed_test_slice = slice(None, reversed_test_index)

    for train_slice, val_slice, test_slice in [
        (
            direct_train_slice,
            direct_val_slice,
            direct_test_slice
        ),
        (
            direct_train_slice,
            direct_test_slice,
            direct_val_slice
        ),
        (
            reversed_train_slice,
            reversed_val_slice,
            reversed_test_slice
        ),
        (
            reversed_train_slice,
            reversed_test_slice,
            reversed_val_slice
        ),
    ]:
        X_train = np.copy(X[train_slice, :])
        Y_train = np.copy(Y[train_slice, :])
        X_val = np.copy(X[val_slice, :])
        Y_val = np.copy(Y[val_slice, :])
        X_test = np.copy(X[test_slice, :])
        Y_test = np.copy(Y[test_slice, :])
        yield X_train, Y_train, X_val, Y_val, X_test, Y_test