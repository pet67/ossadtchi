import scipy
import scipy.io
import scipy.signal
import sklearn
import sklearn.linear_model
import numpy as np

import library.models_lib.common


FEATURES_FREQUENCIES_LIST = list(range(5, 150, 5))
def get_narrowband_features(X, frequency, fir_taps=257, abs_values_high_pass_frequency=0.5):
    assert len(X.shape) == 2
    assert X.shape[0] > X.shape[1]

    b_hp_05, a_hp_05 = scipy.signal.butter(4, abs_values_high_pass_frequency * 1.0 / (frequency / 2), btype='high')
    
    band_filters = []
    for feature_frequency in FEATURES_FREQUENCIES_LIST:
        band_filters.append(scipy.signal.firwin(fir_taps, [0.9 * feature_frequency, 1.1 * feature_frequency], fs=frequency, pass_zero=False))

    X_output = np.zeros((X.shape[0], X.shape[1], len(FEATURES_FREQUENCIES_LIST)))
    for channel in range(X.shape[1]):
        for index, band_filter in enumerate(band_filters):
            new_feature_signal = scipy.signal.filtfilt(band_filter, [1], X[:, channel])
            new_feature_signal = np.absolute(new_feature_signal)
            new_feature_signal = scipy.signal.filtfilt(b_hp_05, a_hp_05, new_feature_signal)
            X_output[:, channel, index] = new_feature_signal
    return X_output


def get_narrowband_features_flat(X, frequency, fir_taps=257, abs_values_high_pass_frequency=0.5):
    return library.models_lib.common.make_flat(get_narrowband_features(X, frequency, fir_taps=fir_taps, abs_values_high_pass_frequency=abs_values_high_pass_frequency))
