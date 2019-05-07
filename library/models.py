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

import keras
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
import math

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
        self.epochs = 1
        self.batch_size = 40
        self.learning_rate = 0.0003
        self.callbacks = [
                keras.callbacks.EarlyStopping(patience=3),
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
