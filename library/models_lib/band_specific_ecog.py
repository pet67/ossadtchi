# Algorithm is based on paper
# Name: "Decoding Finger Flexion from Band-Specific ECoG Signals in Humans"
# Authors "Nanying Liang and Laurent Bougrain"
# Web Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3384842/

import scipy
import scipy.io
import scipy.signal
import numpy as np

import library.models_lib.common


# paper default parameters
FIR_TAPS = 50 # This parameter not mentioned in paper
MOVING_AVERAGE_WINDOW = 41
BACKWARD_NON_OVERLAPPING_PARTS = 26

LAG_BACKWARD = BACKWARD_NON_OVERLAPPING_PARTS * MOVING_AVERAGE_WINDOW
LAG_FORWARD = 0
DECIMATE = MOVING_AVERAGE_WINDOW


def dummy_forward_moving_avarage(signal, window):
    # This is slow version, you may use it for debug 
    new_signal = np.zeros(len(signal))
    for i in range(len(signal)):
        new_signal[i] = sum(signal[i:i+window])
    return new_signal

def forward_moving_avarage(signal, window):
    cumsum_with_padding = np.cumsum(np.pad(signal, (1, window-1), 'constant', constant_values=0))
    return cumsum_with_padding[window:] - cumsum_with_padding[:-window]


def get_band_features(X, frequency, fir_taps=FIR_TAPS, moving_avarage_window=MOVING_AVERAGE_WINDOW):
    band_filters = [
        scipy.signal.firwin(fir_taps, cutoff=[1, 60], fs=frequency, pass_zero=False),
        scipy.signal.firwin(fir_taps, cutoff=[60, 100], fs=frequency, pass_zero=False),
        scipy.signal.firwin(fir_taps, cutoff=[100, 200], fs=frequency, pass_zero=False),  
    ]
    X_output_3D = np.zeros((X.shape[0], X.shape[1], len(band_filters)))
    for channel in range(X.shape[1]):
        for index, band_filter in enumerate(band_filters):
            new_feature_signal = scipy.signal.lfilter(band_filter, [1], X[:, channel])
            new_feature_signal = np.power(new_feature_signal, 2)
            new_feature_signal = forward_moving_avarage(np.copy(new_feature_signal), moving_avarage_window)
            X_output_3D[:, channel, index] = new_feature_signal
    return X_output_3D


def get_band_features_with_lag(X, frequency, lag_backward=LAG_FORWARD, lag_forward=LAG_FORWARD, decimate=DECIMATE):
    return library.models_lib.common.make_lag_3D(get_band_features(X, frequency), lag_backward=lag_backward, lag_forward=lag_forward, decimate=decimate)


def get_band_features_with_lag_flat(X, frequency, lag_backward=LAG_BACKWARD, lag_forward=LAG_FORWARD, decimate=DECIMATE):
    return library.models_lib.common.make_flat(get_band_features_with_lag(X, frequency, lag_backward=lag_backward, lag_forward=lag_forward, decimate=decimate))
