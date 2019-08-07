import copy
import numpy as np
import sklearn
import sklearn.linear_model
from tqdm import tqdm

import tensorflow as tf # noqa
from tensorflow.python.client import device_lib # noqa
config = tf.ConfigProto()  # noqa
config.gpu_options.allow_growth = True  # noqa
sess = tf.Session(config=config)  # noqa

import keras
import keras.optimizers
from keras.layers import Input
from keras.models import Model

import torch
import torch.nn
import torch.autograd

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
    def __init__(self, input_shape, output_shape, frequency, lag_backward, lag_forward):
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
        train_data_generator = library.models_lib.common.data_generator(X_train, Y_train, self.batch_size, self.lag_backward, self.lag_forward)
        test_data_generator = library.models_lib.common.data_generator(X_test, Y_test, self.batch_size, self.lag_backward, self.lag_forward)

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
        full_data_generator = library.models_lib.common.data_generator(X, [], self.batch_size, self.lag_backward, self.lag_forward)
        full_data_steps = library.models_lib.common.calculate_batches_number(X.shape[0], self.total_lag, self.batch_size)
        Y_predicted = self.model.predict_generator(
            generator=full_data_generator,
            steps=full_data_steps
        )
        return Y_predicted

    def slice_target(self, Y):
        return Y[self.lag_backward:-self.lag_forward if self.lag_forward > 0 else None]


class LinearRegressionModel(BenchModel):
    def __init__(self, input_shape, output_shape, frequency, max_number_of_combinations, output_filtration=True):
        self.frequency = frequency
        self.output_filtration = output_filtration
        self.max_number_of_combinations = max_number_of_combinations

    def fit(self, X_train, Y_train, X_test, Y_test):
        self.best_channels_combination = library.models_lib.common.get_best_channels_combination(X_train, Y_train, X_test, Y_test, self.frequency, self.output_filtration, self.max_number_of_combinations)
        X_train_new = library.models_lib.common.get_narrowband_features_flat(X_train[:, self.best_channels_combination], self.frequency)
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(X_train_new, Y_train)

    def predict(self, X):
        X_new = library.models_lib.common.get_narrowband_features_flat(X[:, self.best_channels_combination], self.frequency)
        Y_predicted = self.model.predict(X_new)
        if self.output_filtration:
            Y_predicted_filtered = library.models_lib.common.final_lowpass_filtering(Y_predicted, self.frequency)
        else:
            Y_predicted_filtered = Y_predicted
        return Y_predicted_filtered


class LinearRegressionWithRegularization(BenchModel):
    MODEL = None

    def __init__(self, input_shape, output_shape, frequency, output_filtration):
        assert self.MODEL is not None
        self.frequency = frequency
        self.output_filtration = output_filtration


    def fit(self, X_train, Y_train, X_test, Y_test):
        best_alpha = library.models_lib.common.get_best_alpha(X_train, Y_train, X_test, Y_test, self.frequency, self.MODEL, self.output_filtration)
        self.model = self.MODEL(alpha=best_alpha, fit_intercept=True, normalize=True)
        X_train_new = library.models_lib.common.get_narrowband_features_flat(X_train, self.frequency)
        self.model.fit(X_train_new, Y_train)

    def predict(self, X):
        X_new = library.models_lib.common.get_narrowband_features_flat(X, self.frequency)
        Y_predicted = self.model.predict(X_new)
        if len(Y_predicted.shape) == 1:
            Y_predicted = Y_predicted.reshape((-1, 1))  # Unexpectedly lasso returns (*,) dimension instead (*, 1)
        if self.output_filtration:
            Y_predicted_filtered = library.models_lib.common.final_lowpass_filtering(Y_predicted, self.frequency)
        else:
            Y_predicted_filtered = Y_predicted
        return Y_predicted_filtered


class RidgeRegressionModel(LinearRegressionWithRegularization):
    MODEL = sklearn.linear_model.Ridge


class LassoRegressionModel(LinearRegressionWithRegularization):
    MODEL = sklearn.linear_model.Lasso


class NewSimplePytorchNet(BenchModel):
    def __init__(self, input_shape, output_shape, frequency, lag_backward, lag_forward, output_filtration, best_channels_only):
        assert len(input_shape) == len(output_shape) == 1
        assert lag_backward + lag_forward == 200
        assert frequency == 250 or frequency == 500 or frequency == 1000
        self.iters = 20000
        self.batch_size = 200
        self.learning_rate = 0.0003
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        self.total_lag = self.lag_backward + self.lag_forward
        self.output_filtration = output_filtration
        self.frequency = frequency
        self.best_channels_only = best_channels_only
        self.best_state_dict = None

    def fit(self, X_train, Y_train, X_test, Y_test):
        if self.best_channels_only:
            self.best_channels_combination = library.models_lib.common.get_best_channels_combination(copy.deepcopy(X_train), copy.deepcopy(Y_train), copy.deepcopy(X_test), copy.deepcopy(Y_test), self.frequency, self.output_filtration)
            X_train = X_train[:, self.best_channels_combination]

        self.model = library.models_lib.torch_nets.simple_net(X_train.shape[1], Y_train.shape[1], self.lag_backward, self.lag_forward).cuda()
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_data_generator = library.models_lib.common.data_generator(X_train, Y_train, self.batch_size, self.lag_backward, self.lag_forward)
        Y_test_sliced = self.slice_target(Y_test)
        pbar = tqdm(total=self.iters)
        loss_history = []
        max_test_corr = -1
        for batch_number, (x_batch, y_batch) in enumerate(train_data_generator):
            self.model.train()
            assert x_batch.shape[0] == y_batch.shape[0]
            x_batch = torch.autograd.Variable(torch.FloatTensor(x_batch)).cuda()
            y_batch = torch.autograd.Variable(torch.FloatTensor(y_batch)).cuda()
            self.optimizer.zero_grad()
            y_predicted = self.model(x_batch)
            loss = self.loss_function(y_predicted, y_batch)
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.cpu().data.numpy())
            pbar.update(1)
            eval_lag = min(100, len(loss_history))
            pbar.set_postfix(loss=np.mean(loss_history[-eval_lag:]))
            if len(loss_history) % 250 == 0:
                Y_predicted = self.predict(X_test)
                test_corr_list = []
                for i in range(Y_predicted.shape[1]):
                    test_corr = np.corrcoef(Y_predicted[:, i], Y_test_sliced[:, i], rowvar=False)[0, 1]
                    test_corr_list.append(test_corr)
                mean_corr = np.mean(test_corr_list)
                print("Correlation test:", mean_corr)
                if mean_corr > max_test_corr:
                    max_test_corr = mean_corr
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
            if batch_number >= self.iters:
                break
        pbar.close()
        self.model = library.models_lib.torch_nets.simple_net(X_train.shape[1], Y_train.shape[1], self.lag_backward, self.lag_forward).cuda()
        assert self.best_state_dict is not None
        self.model.load_state_dict(self.best_state_dict)

    def predict(self, X):
        if self.best_channels_only:
            X = X[:, self.best_channels_combination]
        full_data_generator = library.models_lib.common.data_generator(X, [], self.batch_size, self.lag_backward, self.lag_forward)
        full_data_steps = library.models_lib.common.calculate_batches_number(X.shape[0], self.total_lag, self.batch_size)
        Y_predicted = []
        for batch_number, x_batch in enumerate(full_data_generator):
            self.model.eval()
            x_batch = torch.autograd.Variable(torch.FloatTensor(x_batch)).cuda()
            y_predicted = self.model(x_batch).cpu().data.numpy()
            Y_predicted.append(y_predicted)
            if batch_number >= full_data_steps - 1:
                break
        Y_predicted = np.concatenate(Y_predicted, axis=0)
        if self.output_filtration:
            Y_predicted_filtered = library.models_lib.common.final_lowpass_filtering(Y_predicted, self.frequency)
            return Y_predicted_filtered
        else:
            return Y_predicted

    def slice_target(self, Y):
        return Y[self.lag_backward:-self.lag_forward if self.lag_forward > 0 else None]
