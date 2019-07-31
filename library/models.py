import copy
import math

import numpy as np
import scipy
import scipy.signal
import sklearn
import sklearn.linear_model
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.client import device_lib
config = tf.ConfigProto()  # noqa
config.gpu_options.allow_growth = True  # noqa
sess = tf.Session(config=config)  # noqa

import keras
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras import backend as K

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import library.models_lib
import library.models_lib.common
import library.models_lib.keras_nets
import library.models_lib.torch_nets
import library.models_lib.other_models


class BenchModel:
    def __init__(self, input_shape, output_shape, frequency):
        raise NotImplementedError

    def fit(self, X_train, Y_train, X_test, Y_test):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def slice_target(self, Y):
        return Y
    

class Iterative2DRegressionModel(BenchModel):
    def __init__(self, input_shape, output_shape, frequency, output_filtration=True):
        self.frequency = frequency
        self.output_filtration = output_filtration

    def fit(self, X_train, Y_train, X_test, Y_test):
        self.model = library.models_lib.other_models.Iterative2DRegression(self.frequency)
        X_train_3D = library.models_lib.common.get_narrowband_features(X_train, self.frequency)
        X_test_3D = library.models_lib.common.get_narrowband_features(X_test, self.frequency)

        self.model.fit(X_train_3D, Y_train, X_test_3D, Y_test)

    def predict(self, X):
        X_3D = library.models_lib.common.get_narrowband_features(X, self.frequency)
        Y_predicted = self.model.predict(X_3D)
        if self.output_filtration:
            Y_predicted_filtered = library.models_lib.common.final_lowpass_filtering(Y_predicted, self.frequency)
        else:
            Y_predicted_filtered = Y_predicted
        return Y_predicted_filtered


class BaselineNet(BenchModel):
    def __init__(self, input_shape, output_shape, frequency,  lag_backward, lag_forward):
        assert len(input_shape) == len(output_shape) == 1
        assert lag_backward >= 0
        assert lag_forward >= 0
        assert frequency == 2000
        self.number_of_input_channels = input_shape[0]
        self.number_of_output_channels = output_shape[0]
        self.epochs = 40
        self.batch_size = 40
        self.learning_rate = 0.0003
        self.callbacks = [
                keras.callbacks.EarlyStopping(patience=4),
                keras.callbacks.TerminateOnNaN(),
        ]

        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        self.total_lag = self.lag_backward + self.lag_forward

        optimizer = keras.optimizers.Adam(lr=self.learning_rate)

        inputs = Input(shape=(lag_backward + lag_forward + 1, self.number_of_input_channels))
        predictions = library.models_lib.baseline_net.baseline(inputs, self.number_of_output_channels)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer=optimizer, loss='mse')


    def fit(self, X_train, Y_train, X_test, Y_test):
        train_data_generator = data_generator(X_train, Y_train, self.batch_size, self.lag_backward, self.lag_forward)
        test_data_generator = data_generator(X_test, Y_test, self.batch_size, self.lag_backward, self.lag_forward)

        train_data_steps = library.models_lib.common.calculate_batches_number(X_train.shape[0], self.total_lag, self.batch_size)
        test_data_steps = library.models_lib.common.calculate_batches_number(X_test.shape[0], self.total_lag, self.batch_size)

        self.model.fit_generator(
            generator=train_data_generator,
            steps_per_epoch=train_data_steps,
            validation_data=test_data_generator,
            validation_steps=test_data_steps,
            epochs=self.epochs,
            verbose=True,
            callbacks=self.callbacks,
        )

    def predict(self, X):
        full_data_generator = data_generator(X, [], self.batch_size, self.lag_backward, self.lag_forward)
        full_data_steps = library.models_lib.common.calculate_batches_number(X.shape[0], self.total_lag, self.batch_size)
        Y_predicted = self.model.predict_generator(
            generator=full_data_generator,
            steps=full_data_steps
        )
        return Y_predicted

    def slice_target(self, Y):
        return Y[self.lag_backward : -self.lag_forward if self.lag_forward > 0 else None]


class LinearRegressionModel(BenchModel):
    def __init__(self, input_shape, output_shape, frequency, output_filtration=True):
        self.frequency = frequency
        self.output_filtration = output_filtration

    def fit(self, X_train, Y_train, X_test, Y_test):
        self.best_channels_combination = library.models_lib.common.get_best_combination_flat(X_train, Y_train, X_test, Y_test, self.frequency)
        X_train_new = library.models_lib.common.get_narrowband_features_flat(X_train[:, self.best_channels_combination], self.frequency)
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(X_train_new, Y_train)

    def predict(self, X):
        X_new = library.models_lib.common.get_narrowband_features_flat(X_train[:, self.best_channels_combination], self.frequency)
        Y_predicted = self.model.predict(X_new)
        if output_filtration:
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
        else:
            Y_predicted_filtered = Y_predicted
        return Y_predicted_filtered


class LinearRegressionWithRegularization(BenchModel):
    MODEL = None
    def __init__(self, input_shape, output_shape, frequency, output_filtration=True):
        assert self.MODEL is not None
        self.frequency = frequency
        self.output_filtration = output_filtration

    def fit(self, X_train, Y_train, X_test, Y_test):
        best_alpha = library.models_lib.common.get_best_alpha(X_train, Y_train, X_test, Y_test, self.frequency, self.MODEL)
        self.model = self.MODEL(alpha=best_alpha, fit_intercept=True, normalize=True)
        X_train_new = library.models_lib.common.get_narrowband_features(X_train, self.frequency)
        self.model.fit(X_train_new, Y_train)

    def predict(self, X):
        X_new = library.models_lib.common.get_narrowband_features(X, self.frequency)
        Y_predicted = self.model.predict(X_new)
        if len(Y_predicted.shape) == 1:
            Y_predicted = Y_predicted.reshape((-1, 1)) # Unexpectedly lasso returns (*,) dimension instead (*, 1)
        if output_filtration:
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
        else:
            Y_predicted_filtered = Y_predicted
        return Y_predicted_filtered


class RidgeRegressionModel(LinearRegressionWithRegularization):
    MODEL = sklearn.linear_model.Ridge


class LassoRegressionModel(LinearRegressionWithRegularization):
    MODEL = sklearn.linear_model.Lasso


class NewSimplePytorchNet(BenchModel):
    def __init__(self, input_shape, output_shape, frequency, lag_backward, lag_forward, low_pass_filtering, best_channels_only):
        assert len(input_shape) == len(output_shape) == 1
        assert lag_backward == 200
        assert lag_forward == 0
        assert frequency == 250 or frequency == 500 or frequency == 1000
        self.number_of_input_channels = input_shape[0]
        self.number_of_output_channels = output_shape[0]
        self.iters = 20000
        self.batch_size = 200
        self.learning_rate = 0.0003
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        self.total_lag = self.lag_backward + self.lag_forward
        self.low_pass_filtering = low_pass_filtering
        self.frequency = frequency
        self.best_channels_only = best_channels_only
        
        if self.best_channels_only:
            self.bench_model = LinearRegressionModel(input_shape, output_shape, self.frequency)

    def fit(self, X_train, Y_train, X_test, Y_test):
        if self.best_channels_only:
            self.bench_model.fit(copy.deepcopy(X_train), copy.deepcopy(Y_train), copy.deepcopy(X_test), copy.deepcopy(Y_test))
            self.best_channels_combination = self.bench_model.best_channels_combination
            X_train = X_train[:, self.best_channels_combination]

        self.model = simple_net(X_train.shape[1], self.number_of_output_channels, self.lag_backward, self.lag_forward).cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

        train_data_generator = data_generator(X_train, Y_train, self.batch_size, self.lag_backward, self.lag_forward)
        Y_test_sliced = self.slice_target(Y_test)
        pbar = tqdm(total=self.iters)
        loss_history = []
        max_test_corr = 0
        best_iter = 0
        for batch_number, (x_batch, y_batch) in enumerate(train_data_generator):
            self.model.train()
            assert x_batch.shape[0]==y_batch.shape[0]
            x_batch = Variable(torch.FloatTensor(x_batch)).cuda()
            y_batch = Variable(torch.FloatTensor(y_batch)).cuda()
            self.optimizer.zero_grad()
            y_predicted = self.model(x_batch)
            loss = self.loss_function(y_predicted,y_batch)
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.cpu().data.numpy())    
            pbar.update(1)
            eval_lag = min(100,len(loss_history))
            pbar.set_postfix(loss = np.mean(loss_history[-eval_lag:]))
            if len(loss_history) % 500 == 0:
                Y_predicted = self.predict(copy.deepcopy(X_test))
                test_corr_list = []
                for i in range(Y_predicted.shape[1]):
                    test_corr = np.corrcoef(Y_predicted[:, i], Y_test_sliced[:, i], rowvar=False)[0,1]
                    test_corr_list.append(test_corr)
                mean_corr = np.mean(test_corr_list)
                print("Correlation test:", mean_corr)
                if mean_corr > max_test_corr:
                    max_test_corr = mean_corr
                    best_iter = len(loss_history)
                if max_test_corr > 0.3 and mean_corr / max_test_corr < 0.8 or (len(loss_history) - best_iter) > 5000:
                    print("Overfitting finish train")
                    break
            if batch_number >= self.iters:
                break
        pbar.close()

    def predict(self, X):
        if self.best_channels_only:
            X = X[:, self.best_channels_combination]
        full_data_generator = data_generator(X, [], self.batch_size, self.lag_backward, self.lag_forward)
        full_data_steps = library.models_lib.common.calculate_batches_number(X.shape[0], self.total_lag, self.batch_size)
        Y_predicted = []
        for batch_number, x_batch in enumerate(full_data_generator):
            self.model.eval()
            x_batch = Variable(torch.FloatTensor(x_batch)).cuda()
            y_predicted = self.model(x_batch).cpu().data.numpy()
            Y_predicted.append(y_predicted)
            if batch_number >= full_data_steps - 1:
                break
        Y_predicted = np.concatenate(Y_predicted, axis = 0)
        if self.low_pass_filtering:
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
            return Y_predicted_filtered
        else:
            return Y_predicted

    def slice_target(self, Y):
        return Y[self.lag_backward : -self.lag_forward if self.lag_forward > 0 else None]
