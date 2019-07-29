import numpy as np
import sklearn
import sklearn.linear_model
import torch
import torch.nn.functional as F
import sys
from tqdm import tqdm


class Iterative2DRegression:
    ITERATIONS = 2000
    FEATURES_PER_CHANNEL = 29 # Defined by ususal Ossadtchi linear regression pipeline

    def __init__(self, input_shape, frequency, features_per_channel):
        self.channels = input_shape[2]
        self.frequency = frequency
        self.W = np.random.normal(0, 1, self.channels)
        self.alpha_w = 0
        self.V = np.random.normal(0, 1, self.FEATURES_PER_CHANNEL)
        self.alpha_v = 0

    def convolution_and_reshaping(self, X_train_3D, conv_type):
        if conv_type == 'channels':
            final_slice = [slice(None), 0, 0, slice(None)]
            weight = torch.FloatTensor(self.W.reshape((1, 1, self.channels, 1)))
            bias = torch.FloatTensor([self.alpha_w])
        elif conv_type == 'features':
            final_slice = [slice(None), 0, slice(None), 0]
            weight = torch.FloatTensor(self.V.reshape((1, 1, 1, self.FEATURES_PER_CHANNEL)))
            bias = torch.FloatTensor([self.alpha_v])
        else:
            raise ValueError
        return F.conv2d(X_train_3D, weight=weight, bias=bias).numpy()[final_slice]

    def w_step(self, X_train_3D, Y_train):
        WtX = self.convolution_and_reshaping(X_train_3D, 'channels')
        assert WtX.shape[1] == len(self.V), f'{WtX.shape[1]} != {len(self.V)}'
        regression = sklearn.linear_model.LinearRegression().fit(WtX, Y_train)
        assert len(self.V) == len(regression.coef_[0]), f'{len(self.W)} != {len(regression.coef_[0])}'
        self.V = regression.coef_[0]
        self.alpha_v = regression.intercept_[0]

    def v_step(self, X_train_3D, Y_train):
        XV = self.convolution_and_reshaping(X_train_3D, 'features')
        assert XV.shape[1] == len(self.W), f'{XV.shape[1]} != {len(self.W)}'
        regression = sklearn.linear_model.LinearRegression().fit(XV, Y_train)
        assert len(self.W) == len(regression.coef_[0]), f'{len(self.W)} != {len(regression.coef_[0])}'
        self.W = regression.coef_[0]
        self.alpha_w = regression.intercept_[0]

    def fit(self, X_train, Y_train, X_test, Y_test):
        X_train_3D = torch.Tensor(X_train)
        X_test_3D = torch.Tensor(X_test)
        assert X_train_3D.shape[3] == self.FEATURES_PER_CHANNEL, f'{X_train_3D.shape[3]} != {self.FEATURES_PER_CHANNEL}'
        assert X_train_3D.shape[2] == self.channels, f'{X_train_3D.shape[2]} != {self.channels}'

        for iteration in range(self.ITERATIONS):
            self.w_step(X_train_3D, Y_train)
            self.v_step(X_train_3D, Y_train)
            Y_predicted = self.predict(X_test_3D)
            print(iteration, np.corrcoef(Y_predicted, Y_test, rowvar=False)[0,1])

    def predict(self, X):
        X_3D = torch.Tensor(X)
        WtX = self.convolution_and_reshaping(X_3D, 'channels')
        Y_predicted = self.convolution_and_reshaping(torch.Tensor(WtX.reshape((-1, 1, 1, self.FEATURES_PER_CHANNEL))), 'features')
        return Y_predicted