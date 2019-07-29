import tensorflow as tf
from tensorflow.python.client import device_lib
config = tf.ConfigProto()  # noqa
config.gpu_options.allow_growth = True  # noqa
sess = tf.Session(config=config)  # noqa

from tqdm import tqdm

import keras
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras import backend as K

import keras
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
import math

from torch.autograd import Variable
import torch, torch.nn as nn
import torch.nn.functional as F


import copy
import sklearn
import sklearn.linear_model

import scipy
import scipy.signal

import numpy as np

class BenchModel:
    def __init__(self, input_shape, output_shape, frequency):
        raise NotImplementedError

    def fit(self, X_train, Y_train, X_test, Y_test):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def slice_target(self, Y):
        return Y


def data_generator(X, Y, b_size, lag_backward, lag_forward):
    total_lag = lag_backward + lag_forward
    all_b = (X.shape[0] - total_lag) / b_size
    samples_in_last_batch = (X.shape[0] - total_lag) % b_size
    batch = 0
    randome_core = np.asarray(range(lag_backward, X.shape[0] - lag_forward))
    while True:
        if (all_b - batch <= 0 or batch == 0):
            if len(Y) > 0:
                randome_core = np.random.choice(range(lag_backward, X.shape[0] - lag_forward),
                                                size=X.shape[0] - total_lag, replace=False)
            batch = 0
        if (all_b - batch >= 1):
            samplaes_in_batch = b_size
        else:
            samplaes_in_batch = samples_in_last_batch
        batch += 1
        batch_start = (batch - 1) * b_size
        batch_end = batch_start + samplaes_in_batch
        batch_samples = randome_core[batch_start:batch_end]
        batch_x = np.atleast_3d([X[i - lag_backward:i + lag_forward + 1] for i in batch_samples])
        if len(Y) > 0:
            batch_y = Y[randome_core[((batch - 1) * b_size):((batch - 1) * b_size + samplaes_in_batch)]]
            yield (batch_x, batch_y)
        else:
            yield batch_x


def final_lowpass_filtering(Y_predicted, frequency):
    b_lowpass_2Hz, a_lowpass_2Hz = scipy.signal.butter(4, Wn=2 / (frequency / 2), btype='low')
    Y_predicted_filtered = np.copy(Y_predicted)
    for i in range(Y_predicted.shape[1]):
        Y_predicted_filtered[:, i] = scipy.signal.filtfilt(b_lowpass_2Hz, a_lowpass_2Hz, Y_predicted[:, i])
    assert Y_predicted.shape == Y_predicted_filtered.shape
    return Y_predicted_filtered


class envelope_detector(nn.Module):
    def __init__(self, in_channels, channels_per_channel):
        super(self.__class__,self).__init__()
        self.FILTERING_SIZE = 50
        self.ENVELOPE_SIZE = self.FILTERING_SIZE * 2
        self.CHANNELS_PER_CHANNEL = channels_per_channel
        self.OUTPUT_CHANNELS = self.CHANNELS_PER_CHANNEL * in_channels
        self.conv_filtering = nn.Conv1d(in_channels, self.OUTPUT_CHANNELS, bias=False, kernel_size=self.FILTERING_SIZE, groups=in_channels)
        self.conv_envelope = nn.Conv1d(self.OUTPUT_CHANNELS, self.OUTPUT_CHANNELS, kernel_size=self.ENVELOPE_SIZE, groups=self.OUTPUT_CHANNELS)
        
    def forward(self, x):
        x = self.conv_filtering(x)
        x = F.leaky_relu(x, negative_slope=-1)
        x = self.conv_envelope(x)
        return x


class simple_filtering(nn.Module):
    def __init__(self, in_channels):
        super(self.__class__,self).__init__()
        self.SIMPLE_FILTER_SIZE = 149
        self.simple_filter = nn.Conv1d(in_channels, in_channels, bias=False, kernel_size=self.SIMPLE_FILTER_SIZE, groups=in_channels)

    def forward(self, x):
        x = self.simple_filter(x)
        return x

    
class Dropout1D(nn.Module):
    def __init__(self, p):
        super(self.__class__,self).__init__()
        self.dropout_2d = nn.Dropout2d(p)

    def forward(self, x):
        x = x.unsqueeze(2)           # (N, C, 1, T)
        x = self.dropout_2d(x)       # (N, K, 1, T)
        x = x.squeeze(2)             # (N, C, T)
        return x


class simple_net(nn.Module):
    def __init__(self, in_channels, output_channels, lag_backward, lag_forward):
        super(self.__class__,self).__init__()
        self.ICA_CHANNELS = 0
        self.CHANNELS_PER_CHANNEL = 15
        self.OUTPUT_FILTERING = False
        self.EXTRA_FEATURES = False
        
        self.total_input_channels = self.ICA_CHANNELS + in_channels
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        self.middle_point_share = lag_backward * 1.0 / (lag_backward + lag_forward)

        self.final_out_features = (self.CHANNELS_PER_CHANNEL + 2) * self.total_input_channels - 2 * in_channels * int(not self.EXTRA_FEATURES) 

        if self.ICA_CHANNELS > 0:
            self.ica = nn.Conv1d(in_channels, self.ICA_CHANNELS, 1)

        self.detector = envelope_detector(self.total_input_channels, self.CHANNELS_PER_CHANNEL)
        self.simple_filter = simple_filtering(self.total_input_channels)
        self.features_batchnorm = torch.nn.BatchNorm1d(self.final_out_features, affine=False)
        self.unmixed_batchnorm = torch.nn.BatchNorm1d(self.total_input_channels, affine=False)

        if self.OUTPUT_FILTERING:
            self.final_dropout = Dropout1D(p=0.8)
            self.wights_second = nn.Conv1d(self.final_out_features, output_channels, 1)
            self.predictions_filtering = nn.Conv1d(output_channels, output_channels, bias=True, kernel_size=52)
        else:
            self.final_dropout = torch.nn.Dropout(p=0.8)
            self.wights_second = nn.Linear(self.final_out_features, output_channels)



    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2) # for data_generator compareability
        if self.ICA_CHANNELS > 0:
            inputs_unmixed = self.ica(inputs)
            all_inputs = torch.cat((inputs, inputs_unmixed), 1)
            all_inputs = self.unmixed_batchnorm(all_inputs)
        else:
            all_inputs = inputs

        detected_envelopes = self.detector(all_inputs)
        if self.EXTRA_FEATURES:
            simple_filtered_signals = self.simple_filter(all_inputs)
            features = torch.cat((detected_envelopes, simple_filtered_signals, all_inputs[:, :, 148:]), 1)
        else:
            features = detected_envelopes
        
        if self.OUTPUT_FILTERING:
            features = self.features_batchnorm(features)
            features = self.final_dropout(features)
            output = self.wights_second(features)
            output = self.predictions_filtering(output)
            output = output.squeeze(2)
        else:
            middle_point = int((features.shape[-1]- 1) * self.middle_point_share) 
            features  = features[:, :, middle_point]
            features = features.contiguous()

            features = features.view(features.size(0), -1)
            features = self.features_batchnorm(features)
            features = self.final_dropout(features)
            output = self.wights_second(features)

        return output


def baseline(x, output_channels):

    for nb_filters in [10, 25, 35, 50]:
        x = Conv1D(nb_filters, 50, padding="same")(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(4)(x)
        x = SpatialDropout1D(0.5)(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(10)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)

    predictions = Dense(output_channels)(x)

    return predictions

def calculate_batches_number(x_size, total_lag, b_size):
    return math.ceil((x_size - total_lag) / b_size)


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
        predictions = baseline(inputs, self.number_of_output_channels)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer=optimizer, loss='mse')



    def fit(self, X_train, Y_train, X_test, Y_test):
        train_data_generator = data_generator(X_train, Y_train, self.batch_size, self.lag_backward, self.lag_forward)
        test_data_generator = data_generator(X_test, Y_test, self.batch_size, self.lag_backward, self.lag_forward)

        train_data_steps = calculate_batches_number(X_train.shape[0], self.total_lag, self.batch_size)
        test_data_steps = calculate_batches_number(X_test.shape[0], self.total_lag, self.batch_size)

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
        full_data_steps = calculate_batches_number(X.shape[0], self.total_lag, self.batch_size)
        Y_predicted = self.model.predict_generator(
            generator=full_data_generator,
            steps=full_data_steps
        )
        return Y_predicted

    def slice_target(self, Y):
        return Y[self.lag_backward : -self.lag_forward if self.lag_forward > 0 else None]


class LinearModel(BenchModel):
    def __init__(self, input_shape, output_shape, frequency):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.frequency = frequency
        self.model = None

    def get_features(self, X, channels):
        NumFIRTaps = 257
        cutoff_frequency_for_abs_values = 0.5
        f0 = [0.5, 1, 2, 3, 4] + list(range(5, 150, 5))
        number_of_filters = len(f0)
        b = np.zeros((number_of_filters, NumFIRTaps))
        for i, f in enumerate(f0):
            b[i, :] = scipy.signal.firwin(NumFIRTaps, [0.9 * f / (self.frequency / 2), 1.1 * f / (self.frequency / 2)], pass_zero=False)
        X_new = []
        for channel in channels:
            X_new_single = np.zeros((X.shape[0], len(f0)))
            for i in range(number_of_filters):
                X_new_single[:, i] = scipy.signal.filtfilt(b[i, :], [1], X[:, channel])
            X_new_single = np.absolute(X_new_single)
            b_hp_05, a_hp_05 = scipy.signal.butter(4, cutoff_frequency_for_abs_values / (self.frequency / 2), btype='high')
            for i in range(X_new_single.shape[1]):
                X_new_single[:, i] = scipy.signal.filtfilt(b_hp_05, a_hp_05, X_new_single[:, i])
            X_new.append(X_new_single)
        return np.concatenate(X_new, axis=1)


class LinearRegressionModel(LinearModel):
    MAX_NUMBER_OF_COMBINATIONS = 5

    def __init__(self, input_shape, output_shape, frequency):
        super(LinearRegressionModel, self).__init__(input_shape, output_shape, frequency)
        self.best_channels_combination = None

    def get_results_for_all_channels(self, X_train, Y_train, X_test, Y_test):
        results = {}
        for channel in range(X_train.shape[1]):
            model = sklearn.linear_model.LinearRegression()
            X_train_new = self.get_features(X_train, [channel])
            model.fit(X_train_new, Y_train)
            X_test_new = self.get_features(X_test, [channel])
            Y_predicted = model.predict(X_test_new)
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
            test_corr = np.corrcoef(Y_predicted_filtered.reshape(1, -1), Y_test.reshape(1, -1))[0, 1]
            print(f"Channel {channel}: {round(test_corr, 2)} correlation")
            results[channel] = test_corr
        return results
    
    def get_best_combination(self, X_train, Y_train, X_test, Y_test, best_channels_list):
        max_corr = -1
        best_channels_combination = None
        for channels_number in range(1, self.MAX_NUMBER_OF_COMBINATIONS + 1):
            model = sklearn.linear_model.LinearRegression()
            X_train_new = self.get_features(X_train, best_channels_list[:channels_number])
            model.fit(X_train_new, Y_train)
            X_test_new = self.get_features(X_test, best_channels_list[:channels_number])
            Y_predicted = model.predict(X_test_new)
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
            test_corr = np.corrcoef(Y_predicted_filtered.reshape(1, -1), Y_test.reshape(1, -1))[0, 1]
            if test_corr > max_corr:
                best_channels_combination = best_channels_list[:channels_number]
                max_corr = test_corr
            print(f"Channes {best_channels_list[:channels_number]}: {round(test_corr, 2)} correlation")
        return best_channels_combination

    def fit(self, X_train, Y_train, X_test, Y_test):
        results = self.get_results_for_all_channels(X_train, Y_train, X_test, Y_test)
        best_channels_list = [i[0] for i in sorted(results.items(), key=lambda x: x[1], reverse=True)]
        self.best_channels_combination = self.get_best_combination(X_train, Y_train, X_test, Y_test, best_channels_list)
        X_train_new = self.get_features(X_train, self.best_channels_combination)
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(X_train_new, Y_train)

    def predict(self, X):
        X_new = self.get_features(X, self.best_channels_combination)
        Y_predicted = self.model.predict(X_new)
        Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
        return Y_predicted_filtered


class RidgeRegressionModel(LinearModel):
    def __init__(self, input_shape, output_shape, frequency):
        super(RidgeRegressionModel, self).__init__(input_shape, output_shape, frequency)
        self.all_channels_list = range(input_shape[0])

    def get_best_alpha(self, X_train, Y_train, X_test, Y_test):
        max_corr = -1
        best_alpha = None
        X_train_new = self.get_features(X_train, self.all_channels_list)
        X_test_new = self.get_features(X_test, self.all_channels_list)
        for alpha in [20, 10, 7, 5, 4, 2, 1, 0.1, 0.01, 0.001, 0.0001]:
            model = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=True, normalize=True)
            model.fit(X_train_new, Y_train)
            Y_predicted = model.predict(X_test_new)
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
            test_corr = np.corrcoef(Y_predicted_filtered.reshape(1, -1), Y_test.reshape(1, -1))[0, 1]
            if test_corr > max_corr:
                best_alpha = alpha
                max_corr = test_corr
            print(f"Alpha {alpha}: {round(test_corr, 2)} correlation")
        return best_alpha

    def fit(self, X_train, Y_train, X_test, Y_test):
        best_alpha = self.get_best_alpha(X_train, Y_train, X_test, Y_test)
        self.model = sklearn.linear_model.Ridge(alpha=best_alpha, fit_intercept=True, normalize=True)
        X_train_new = self.get_features(X_train, self.all_channels_list)
        self.model.fit(X_train_new, Y_train)

    def predict(self, X):
        X_new = self.get_features(X, self.all_channels_list)
        Y_predicted = self.model.predict(X_new)
        Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
        return Y_predicted_filtered


class LassoRegressionModel(LinearModel):
    def __init__(self, input_shape, output_shape, frequency):
        super(LassoRegressionModel, self).__init__(input_shape, output_shape, frequency)
        self.all_channels_list = range(input_shape[0])

    def get_best_alpha(self, X_train, Y_train, X_test, Y_test):
        max_corr = -1
        best_alpha = None
        X_train_new = self.get_features(X_train, self.all_channels_list)
        X_test_new = self.get_features(X_test, self.all_channels_list)
        for alpha in [20, 10, 7, 5, 4, 2, 1, 0.1, 0.01, 0.001, 0.0001]:
            model = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=True, normalize=True)
            model.fit(X_train_new, Y_train)
            Y_predicted = model.predict(X_test_new)
            if len(Y_predicted.shape) == 1:
                Y_predicted = Y_predicted.reshape((-1, 1)) # Unexpectedly lasso returns (*,) dimension instead (*, 1)
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
            test_corr = np.corrcoef(Y_predicted_filtered.reshape(1, -1), Y_test.reshape(1, -1))[0, 1]
            if test_corr > max_corr:
                best_alpha = alpha
                max_corr = test_corr
            print(f"Alpha {alpha}: {round(test_corr, 2)} correlation")
        return best_alpha

    def fit(self, X_train, Y_train, X_test, Y_test):
        best_alpha = self.get_best_alpha(X_train, Y_train, X_test, Y_test)
        X_train_new = self.get_features(X_train, self.all_channels_list)
        self.model = sklearn.linear_model.Lasso(alpha=best_alpha, fit_intercept=True, normalize=True)
        self.model.fit(X_train_new, Y_train)


    def predict(self, X):
        X_new = self.get_features(X, self.all_channels_list)
        Y_predicted = self.model.predict(X_new)
        if len(Y_predicted.shape) == 1:
            Y_predicted = Y_predicted.reshape((-1, 1)) # Unexpectedly lasso returns (*,) dimension instead (*, 1)
        Y_predicted_filtered = final_lowpass_filtering(Y_predicted, self.frequency)
        return Y_predicted_filtered

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
        full_data_steps = calculate_batches_number(X.shape[0], self.total_lag, self.batch_size)
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