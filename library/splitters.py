import numpy as np


def new_4x_cross_testing(X, Y):
    train_share = 0.6
    val_share = (1 - train_share) / 2
    test_share = (1 - train_share) / 2
    assert (train_share + val_share + test_share) == 1

    samples_in_data = int(X.shape[0])
    
    for train_slice, val_slice, test_slice in [
        (
            slice(None, int(samples_in_data * train_share)),
            slice(int(samples_in_data * train_share), int(samples_in_data * (train_share + val_share))),
            slice(int(samples_in_data * (train_share + val_share)), None)
        ),
        (
            slice(None, int(samples_in_data * train_share)),
            slice(int(samples_in_data * (train_share + val_share)), None),
            slice(int(samples_in_data * train_share), int(samples_in_data * (train_share + val_share)))
        ),
        (
            slice(int(samples_in_data * (test_share + val_share)), None),
            slice(int(samples_in_data * test_share), int(samples_in_data * (test_share + val_share))),
            slice(None, int(samples_in_data * test_share))
        ),
        (
            slice(int(samples_in_data * (test_share + val_share)), None),
            slice(None, int(samples_in_data * test_share)),
            slice(int(samples_in_data * test_share), int(samples_in_data * (test_share + val_share)))
        ),
    ]:
        X_train = np.copy(X[train_slice, :])
        Y_train = np.copy(Y[train_slice, :])
        X_val = np.copy(X[val_slice, :])
        Y_val = np.copy(Y[val_slice, :])
        X_test = np.copy(X[test_slice, :])
        Y_test = np.copy(Y[test_slice, :])
        yield X_train, Y_train, X_val, Y_val, X_test, Y_test