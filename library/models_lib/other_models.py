import sys

import numpy as np
import sklearn
import sklearn.linear_model
import torch

import library.models_lib.common


class Iterative2DRegression:
    def __init__(self, frequency, iteraions=200):
        self.frequency = frequency
        self.iteraions = iteraions
        self.best_W = None
        self.best_alpha_w = None
        self.best_V = None
        self.best_alpha_v = None

    def init_random_weights(self,):
        self.W = np.random.normal(0, 1, self.channels)
        self.alpha_w = 0
        self.V = np.random.normal(0, 1, self.features_per_channel)
        self.alpha_v = 0

    def convolution_and_reshaping(self, X_train_3D, conv_type):
        if conv_type == 'channels':
            final_slice = [slice(None), 0, 0, slice(None)]
            weight = torch.FloatTensor(self.W.reshape((1, 1, self.channels, 1)))
            bias = torch.FloatTensor([self.alpha_w])
        elif conv_type == 'features':
            final_slice = [slice(None), 0, slice(None), 0]
            weight = torch.FloatTensor(self.V.reshape((1, 1, 1, self.features_per_channel)))
            bias = torch.FloatTensor([self.alpha_v])
        else:
            raise ValueError
        return torch.nn.functional.conv2d(X_train_3D, weight=weight, bias=bias).numpy()[final_slice]

    def v_step(self, X_train_3D, Y_train):
        WtX = self.convolution_and_reshaping(X_train_3D, 'channels')
        assert WtX.shape[1] == len(self.V), f'{WtX.shape[1]} != {len(self.V)}'
        regression = sklearn.linear_model.LinearRegression().fit(WtX, Y_train)
        assert len(self.V) == len(regression.coef_[0]), f'{len(self.W)} != {len(regression.coef_[0])}'
        self.V = regression.coef_[0]
        self.alpha_v = regression.intercept_[0]

    def w_step(self, X_train_3D, Y_train):
        XV = self.convolution_and_reshaping(X_train_3D, 'features')
        assert XV.shape[1] == len(self.W), f'{XV.shape[1]} != {len(self.W)}'
        regression = sklearn.linear_model.LinearRegression().fit(XV, Y_train)
        assert len(self.W) == len(regression.coef_[0]), f'{len(self.W)} != {len(regression.coef_[0])}'
        self.W = regression.coef_[0]
        self.alpha_w = regression.intercept_[0]

    def get_expanded_channels_dim_tensor(self, X):
        return torch.Tensor(np.expand_dims(X, axis=1))

    def update_best_params(self,):
        self.best_W = self.W
        self.best_alpha_w = self.alpha_w
        self.best_V = self.V
        self.best_alpha_v = self.alpha_v

    def apply_best_params(self,):
        assert all([param is not None for param in [self.best_W, self.best_alpha_w, self.best_V, self.best_alpha_v]])
        self.W = self.best_W
        self.alpha_w = self.best_alpha_w
        self.V = self.best_V
        self.alpha_v = self.best_alpha_v

    def fit(self, X_train_3D, Y_train, X_test_3D, Y_test):
        assert len(X_train_3D.shape) == len(X_test_3D.shape) == 3, f"{X_train_3D.shape}, {X_test_3D.shape}"
        self.channels = X_train_3D.shape[1]
        self.features_per_channel = X_train_3D.shape[2]

        X_train_3D = self.get_expanded_channels_dim_tensor(X_train_3D)
        X_test_3D = self.get_expanded_channels_dim_tensor(X_test_3D)

        assert len(X_train_3D.shape) == len(X_test_3D.shape) == 4, f"{X_train_3D.shape}, {X_test_3D.shape}"
        assert X_train_3D.shape[1:] == X_test_3D.shape[1:]
        assert X_train_3D.shape[1] == 1, f'{X_train_3D.shape[1]} != 1'
        assert X_train_3D.shape[2] == self.channels, f'{X_train_3D.shape[2]} != {self.channels}'
        assert X_train_3D.shape[3] == self.features_per_channel, f'{X_train_3D.shape[3]} != {self.features_per_channel}'

        self.init_random_weights()

        best_corr = -1
        for iteration in range(self.iteraions):
            self.w_step(X_train_3D, Y_train)
            self.v_step(X_train_3D, Y_train)

            Y_predicted = self.predict_inner(X_test_3D)
            Y_predicted_filtered = library.models_lib.common.final_lowpass_filtering(Y_predicted, self.frequency)
            corr_before_filtration = np.corrcoef(Y_predicted, Y_test, rowvar=False)[0, 1]
            corr_after_filtration = np.corrcoef(Y_predicted_filtered, Y_test, rowvar=False)[0, 1]
            corr = max(corr_before_filtration, corr_after_filtration)
            if corr > best_corr:
                best_corr = corr
                self.update_best_params()
            sys.stderr.write(f"{iteration}:\t{round(corr_after_filtration, 2)}\t{round(corr_before_filtration, 2)}\t{round(best_corr, 2)}\n")

        self.apply_best_params()

    def predict_inner(self, X_3D):
        WtX = self.convolution_and_reshaping(X_3D, 'channels')
        WtX = torch.Tensor(WtX.reshape((-1, 1, 1, self.features_per_channel)))
        Y_predicted = self.convolution_and_reshaping(WtX, 'features')
        return Y_predicted

    def predict(self, X_3D, expand_dims=True):
        X_3D = self.get_expanded_channels_dim_tensor(X_3D)
        return self.predict_inner(X_3D)
