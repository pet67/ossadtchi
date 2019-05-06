import numpy as np
import keras
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D


class BenchModel:
    def __init__(self, input_shape, output_shape):
        raise NotImplementedError

    def fit(self, X_train, Y_train, X_test, Y_test):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def slice_target(self, Y):
        return Y


class BaselineNet(BenchModel):
    def __init__(self, input_shape, output_shape, lag_backward, lag_foreward):
        assert len(input_shape) == len(output_shape) == 1
        assert lag_backward >= 0
        assert lag_foreward >= 0
        self.number_of_input_channels = input_shape[0]
        self.number_of_output_channels = output_shape[0]
        self.epochs = 20
        self.batch_size = 40
        self.learning_rate = 0.0003
        # restore_best_weights=True may lead to inflated results
        callbacks = [
                keras.callbacks.EarlyStopping(patience=self.epochs, restore_best_weights=True),
                keras.callbacks.TerminateOnNaN(),
        ]

        self.lag_backward = lag_backward
        self.lag_foreward = lag_foreward
        
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)

        inputs = Input(shape=(lag_backward + lag_forward + 1, self.number_of_input_channels))
        predictions = self.baseline(inputs, self.number_of_output_channels)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer=optimizer, loss='mse')


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

    def fit(self, X_train, Y_train, X_test, Y_test):
        self.model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks
        )

    def predict(self, X):
        raise self.model.predict(X)

    def slice_target(self, Y):
        return Y[self.lag_backward : -self.lag_foreward if self.lag_foreward > 0 else None]
