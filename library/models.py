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

            
            
class envelope_detector(nn.Module):
    def __init__(self, in_channels):
        super(self.__class__,self).__init__()
        self.FILTERING_SIZE = 50
        self.ENVELOPE_SIZE = self.FILTERING_SIZE * 2
        self.CHANNELS_PER_CHANNEL = 5
        self.OUTPUT_CHANNELS = self.CHANNELS_PER_CHANNEL * in_channels
        self.conv_filtering = nn.Conv1d(in_channels, self.OUTPUT_CHANNELS, kernel_size=self.FILTERING_SIZE, groups=in_channels)
        self.conv_envelope = nn.Conv1d(self.OUTPUT_CHANNELS, self.OUTPUT_CHANNELS, kernel_size=self.ENVELOPE_SIZE, groups=in_channels)
        
    def forward(self, x):
        x = self.conv_filtering(x)
        x = F.leaky_relu(x, negative_slope=-1)
        x = self.conv_envelope(x)
        return x


class simple_filtering(nn.Module):
    def __init__(self, in_channels):
        super(self.__class__,self).__init__()
        self.SIMPLE_FILTER_SIZE = 149
        self.simple_filter = nn.Conv1d(in_channels, in_channels, kernel_size=self.SIMPLE_FILTER_SIZE, groups=in_channels)

    def forward(self, x):
        x = self.simple_filter(x)
        return x


class simple_net(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(self.__class__,self).__init__()
        self.ICA_CHANNELS = 10
        self.ica = nn.Conv1d(in_channels, self.ICA_CHANNELS, 1)
        self.detector = envelope_detector(self.ICA_CHANNELS)
        self.simple_filter = simple_filtering(self.ICA_CHANNELS)
        self.channels_batchnorm = torch.nn.BatchNorm1d(70)
        self.spartial_dropout = nn.Dropout2d(p=0.5)
        self.merge_conv = nn.Conv1d(70, 10, 1)
        self.wights = nn.Linear(530, 100)
        self.features_batchnorm = torch.nn.BatchNorm1d(100)
        self.wights_second = nn.Linear(100, output_channels)
        self.final_dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2) 
        inputs_unmixed = self.ica(inputs)
        detected_envelopes = self.detector(inputs_unmixed)
        simple_filtered_signals = self.simple_filter(inputs_unmixed)
        feature_signals = torch.cat((detected_envelopes, simple_filtered_signals, inputs_unmixed[:, :, 74:-74]), 1)
        #feature_signals = self.channels_batchnorm(feature_signals)
        feature_signals = self.spartial_dropout(feature_signals)
        features = self.merge_conv(feature_signals)
        features = features.view(features.size(0), -1)
        features = self.final_dropout(features)
        features = self.wights(features)
        #features = self.features_batchnorm(features)
        #features = self.relu(features)
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


class ExperementPytorchNet(BenchModel):
    def __init__(self, input_shape, output_shape, frequency,  lag_backward, lag_forward):
        assert len(input_shape) == len(output_shape) == 1
        assert lag_backward >= 0
        assert lag_forward >= 0
        assert lag_backward + lag_forward == 200
        assert frequency == 250
        self.number_of_input_channels = input_shape[0]
        self.number_of_output_channels = output_shape[0]
        self.iters = 10000
        self.batch_size = 40
        self.learning_rate = 0.0005
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        self.total_lag = self.lag_backward + self.lag_forward
        
        self.model = simple_net(self.number_of_input_channels, self.number_of_output_channels).cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)


    def fit(self, X_train, Y_train, X_test, Y_test):
        train_data_generator = data_generator(X_train, Y_train, self.batch_size, self.lag_backward, self.lag_forward)
        pbar = tqdm(total=self.iters)
        loss_history = []
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
            if batch_number >= self.iters:
                break
        pbar.close()

    def predict(self, X):
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
        return Y_predicted

    def slice_target(self, Y):
        return Y[self.lag_backward : -self.lag_forward if self.lag_forward > 0 else None]
