import math

import numpy as np
import scipy
import scipy.signal
import sklearn
import sklearn.linear_model


FILEINDEX2TARGET = {
    1: 5,
    2: 6,
    3: 7,
    4: 8,
    5: 9,
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    10: 4
}


def filename2target(filname):
    file_index = int(filname.split("_")[0])
    return FILEINDEX2TARGET[file_index]
    

def path2target(path):
    return filename2target(path.split("/")[-1])


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


def get_narrowband_features(X, frequency):
    assert len(X.shape) == 2
    assert X.shape[0] > X.shape[1]
    NumFIRTaps = 257
    cutoff_frequency_for_abs_values = 0.5
    f0 = [0.5, 1, 2, 3, 4] + list(range(5, 150, 5))
    number_of_filters = len(f0)
    b_hp_05, a_hp_05 = scipy.signal.butter(4, cutoff_frequency_for_abs_values / (frequency / 2), btype='high')
    b = np.zeros((number_of_filters, NumFIRTaps))
    for i, f in enumerate(f0):
        b[i, :] = scipy.signal.firwin(NumFIRTaps, [0.9 * f / (frequency / 2), 1.1 * f / (frequency / 2)], pass_zero=False)
    X_new = []
    for channel in range(X.shape[1]):
        X_new_single = np.zeros((X.shape[0], len(f0)))
        for i in range(number_of_filters):
            X_new_single[:, i] = scipy.signal.filtfilt(b[i, :], [1], X[:, channel])
        X_new_single = np.absolute(X_new_single)
        for i in range(X_new_single.shape[1]):
            X_new_single[:, i] = scipy.signal.filtfilt(b_hp_05, a_hp_05, X_new_single[:, i])
        X_new.append(X_new_single)

    X_output = np.transpose(np.array(X_new), [1, 0, 2])
    return X_output


def make_flat(X_3D):
    return X_3D.reshape(X_3D.shape[0], -1)

def get_narrowband_features_flat(X, frequency):
    X_3D = get_narrowband_features(X, frequency)
    return make_flat(X_3D)


def get_best_channels_combination(X_train, Y_train, X_test, Y_test, frequency, output_filtration=True, model_class=sklearn.linear_model.LinearRegression, max_number_of_combinations=5):
    print("Get best channels")
    results = {}
    X_train_new = get_narrowband_features(X_train, frequency)
    X_test_new = get_narrowband_features(X_test, frequency)
    for channel in range(X_train.shape[1]):
        model = model_class()
        model.fit(make_flat(X_train_new[:, [channel], :]), Y_train)
        Y_predicted = model.predict(make_flat(X_test_new[:, [channel], :]))
        Y_predicted_filtered = final_lowpass_filtering(Y_predicted, frequency)
        test_corr = np.corrcoef(Y_predicted_filtered.reshape(1, -1), Y_test.reshape(1, -1))[0, 1]
        print(f"Channel {channel}: {round(test_corr, 2)} correlation")
        results[channel] = test_corr

    best_channels_list = [i[0] for i in sorted(results.items(), key=lambda x: x[1], reverse=True)]

    max_corr = -1
    best_channels_combination = None
    for channels_number in range(1, max_number_of_combinations + 1):
        model = model_class()
        model.fit(make_flat(X_train_new[:, best_channels_list[:channels_number], :]), Y_train)
        Y_predicted = model.predict(make_flat(X_test_new[:, best_channels_list[:channels_number], :]))
        if output_filtration:
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, frequency)
        else:
            Y_predicted_filtered = Y_predicted
        test_corr = np.corrcoef(Y_predicted_filtered.reshape(1, -1), Y_test.reshape(1, -1))[0, 1]
        if test_corr > max_corr:
            best_channels_combination = best_channels_list[:channels_number]
            max_corr = test_corr
        print(f"Channes {best_channels_list[:channels_number]}: {round(test_corr, 2)} correlation")
    print(f"best_channels_combination {best_channels_combination}")
    return best_channels_combination


def get_best_alpha(X_train, Y_train, X_test, Y_test, frequency, model_class, output_filtration=True):
    max_corr = -1
    best_alpha = None
    X_train_new = get_narrowband_features(X_train, frequency)
    X_test_new = get_narrowband_features(X_test, frequency)
    for alpha in [20, 10, 7, 5, 4, 2, 1, 0.1, 0.01, 0.001, 0.0001]:
        model = model_class(alpha=alpha, fit_intercept=True, normalize=True)
        model.fit(X_train_new, Y_train)
        Y_predicted = model.predict(X_test_new)
        if output_filtration:
            Y_predicted_filtered = final_lowpass_filtering(Y_predicted, frequency)
        else:
            Y_predicted_filtered = Y_predicted
        test_corr = np.corrcoef(Y_predicted_filtered.reshape(1, -1), Y_test.reshape(1, -1))[0, 1]
        if test_corr > max_corr:
            best_alpha = alpha
            max_corr = test_corr
        print(f"Alpha {alpha}: {round(test_corr, 2)} correlation")
    return best_alpha


def calculate_batches_number(x_size, total_lag, b_size):
    return math.ceil((x_size - total_lag) / b_size)
