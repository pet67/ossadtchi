import sklearn
import sklearn.preprocessing

import scipy
import scipy.signal


def change_target_to_speed(X, Y, frequency):
    Y = Y[1:] - Y[:-1]
    X = X[1:]
    return X, Y, frequency


def signals_and_targets_standart_scaling(X, Y, frequency):
    X = sklearn.preprocessing.scale(X)
    Y = sklearn.preprocessing.scale(Y)
    return X, Y, frequency


def downsample(X, Y, frequency, x_times):
    assert x_times in range(13)
    X = scipy.signal.decimate(X, x_times, axis=0)
    Y = scipy.signal.decimate(Y, x_times, axis=0)
    return X, Y, frequency / x_times


def high_pass_filtering(X, Y, frequency, cutoff_frequency):
    b, a = scipy.signal.butter(3, cutoff_frequency / (frequency / 2), 'high')
    for i in range(X.shape[1]):
        X[:, i] = scipy.signal.filtfilt(b, a, X[:, i])
    return X, Y, frequency
