import sklearn
import sklearn.preprocessing

import scipy
import scipy.signal


def signals_and_targets_standart_scaling(X, Y, frequency):
    X = sklearn.preprocessing.scale(X)
    Y = sklearn.preprocessing.scale(Y)
    return X, Y, frequency


def downsample(X, Y, frequency, x_times):
    assert x_times in range(20)
    X = X[::x_times]
    Y = Y[::x_times]
    return X, Y, frequency / x_times


def high_pass_filtering(X, Y, frequency, cutoff_frequency):
    b, a = scipy.signal.butter(3, cutoff_frequency / (frequency / 2), 'high')
    for i in range(X.shape[1]):
        X[:, i] = scipy.signal.filtfilt(b, a, X[:, i])
    return X, Y, frequency
