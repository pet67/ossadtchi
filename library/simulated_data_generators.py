import numpy as np
import scipy
import scipy.signal
import random
import scipy
import math

RANDOM_SEED = 67


def generate_random_filters(number_of_filters, lower_frequency, upper_frequency, filters_broadband, frequency):
    filters_lower_bands = np.random.uniform(lower_frequency, upper_frequency, filters_broadband)
    filters_upper_bands = [min(lower_band + filters_broadband, int(frequency / 2)) for lower_band in filters_lower_bands]
    assert(len(filters_lower_bands) == len(filters_upper_bands))
    filters = []
    for filter_lower_band, filter_upper_band in zip(filters_lower_bands, filters_upper_bands):
        single_filter = scipy.signal.firwin(34, [filter_lower_band, filter_upper_band], fs=frequency, pass_zero = False)
    return filters


def filter_signals(signals, filters):
    number_of_signals = signals.shape[1]
    assert number_of_signals == len(filters)
    filtered_signals = np.copy(signals)
    for index, single_filter in enumerate(filters):
        filtered_signals[:, index] = np.convolve(signals[:, index], single_filter, mode = "same")
    return filtered_signals


def envelope_signals(signals):
    enveloped_signals = []
    for i in range(len(signals)):
        enveloped_signal = np.abs(scipy.signal.hilbert(signals[:, i]))
        enveloped_signals.append(enveloped_signal)
    enveloped_signals = np.array(enveloped_signals)
    return enveloped_signals


def mix_signals(signals, output_dimension):
    input_dimension = signals.shape[1]
    mixing_matrix = np.random.uniform(0, 1, (input_dimension, output_dimension))
    mixed_signals = np.matmul(signals, mixing_matrix)
    return mixed_signals


def mix_signals_with_lag(signals, lag_backward, lag_foreward, output_dimensions):
    mixed_signals = 0
    return mixed_signals



def mixed_random_noise_to_unmixed_random_noise(signals_length, input_dimension, output_dimension, frequency):
    np.random.seed(RANDOM_SEED)
    signals = np.random.normal(0, 1, (signals_length, input_dimension))
    mixed_signals = mix_signals(signals, output_dimension)
    X = mixed_signals
    Y = signals
    return X, Y


def random_noise_to_filtered_random_noise(signals_length, input_dimension, lower_frequency, upper_frequency, filters_broadband, frequency):
    np.random.seed(RANDOM_SEED)
    signals = np.random.normal(0, 1, (signals_length, input_dimension))
    filters = generate_random_filters(input_dimension, lower_frequency, upper_frequency, filters_broadband, frequency)    
    filtered_signals = filter_signals(signals, filters)
    X = signals
    Y = filtered_signals    
    return X, Y


def random_noise_to_filtered_enveloped_random_noise(signals_length, input_dimension, lower_frequency, upper_frequency, filters_broadband, frequency):
    np.random.seed(RANDOM_SEED)
    signals = np.random.normal(0, 1, (signals_length, input_dimension))
    filters = generate_random_filters(input_dimension, lower_frequency, upper_frequency, filters_broadband, frequency)    
    filtered_signals = filter_signals(signals, filters)
    enveloped_signals = envelope_signals(filtered_signals)
    X = signals
    Y = enveloped_signals    
    return X, Y


def mixed_random_noise_to_unmixed_filtered_random_noise(signals_length, input_dimension, output_dimension, lower_frequency, upper_frequency, filters_broadband, frequency):
    np.random.seed(RANDOM_SEED)
    signals = np.random.normal(0, 1, (signals_length, input_dimension))
    mixed_signals = mix_signals(signals, output_dimension)
    filters = generate_random_filters(input_dimension, lower_frequency, upper_frequency, filters_broadband, frequency)    
    filtered_signals = filter_signals(signals, filters)
    X = mixed_signals
    Y = filtered_signals    
    return X, Y


def mixed_random_noise_to_unmixed_filtered_enveloped_random_noise(signals_length, input_dimension, output_dimension, lower_frequency, upper_frequency, filters_broadband, frequency):
    np.random.seed(RANDOM_SEED)
    signals = np.random.normal(0, 1, (signals_length, input_dimension))
    mixed_signals = mix_signals(signals, output_dimension)
    filters = generate_random_filters(input_dimension, lower_frequency, upper_frequency, filters_broadband, frequency)    
    filtered_signals = filter_signals(signals, filters)
    enveloped_signals = envelope_signals(filtered_signals)
    X = mixed_signals
    Y = enveloped_signals    
    return X, Y
