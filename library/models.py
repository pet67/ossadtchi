class Model:
    def __init__(self, **kwargs):
        raise NotImplementedError

    def fit(self, X_train, Y_train):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def slice(self, Y):
        return Y
