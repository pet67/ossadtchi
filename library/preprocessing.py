import sklearn
import sklearn.preprocessing


def signals_and_targets_standart_scaling(X, Y, frequency):
    X = sklearn.preprocessing.scale(X)
    Y = sklearn.preprocessing.scale(Y)
    return X, Y, frequency
