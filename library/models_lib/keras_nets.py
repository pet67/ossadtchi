import keras
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D


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