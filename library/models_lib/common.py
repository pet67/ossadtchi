import math
import copy

import numpy as np
import scipy
import scipy.signal
import sklearn
import sklearn.linear_model


def make_flat(X_3D):
    return X_3D.reshape(X_3D.shape[0], -1)

def calculate_batches_number(x_size, total_lag, b_size):
    return math.ceil((x_size - total_lag) / b_size)


def make_lag_3D(X_3D, lag_backward, lag_forward, decimate=1):
    assert decimate > 0
    assert lag_backward >=0
    assert lag_forward >=0
    if lag_backward == 0 and lag_forward == 0:
        return X_3D
    X_output_3D = np.zeros(X.shape[0] - lag_backward - lag_forward, X.shape[1], X.shape[2] * (1 + int(lag_backward / decimate) + int(lag_forward / decimate)))
    for i in range(1, lag_backward + 1, decimate):
        X_output_3D[:, :, i] = X_3D[lag_backward - i:-lag_forward - i]
    X_output_3D[:, :, int(lag_backward / decimate)] = X_3D[lag_backward:-lag_forward if lag_forward > 0 else None]  # cetntral point
    for i in range(1, lag_forward + 1, decimate):
        X_output_3D[:, :, i] = X_3D[lag_backward + i:-lag_forward + i if lag_forward - i > 0 else None]
    return X_output_3D


def final_lowpass_filtering(Y_predicted, frequency, lowpass_frequency=2):
    b_lowpass_2Hz, a_lowpass_2Hz = scipy.signal.butter(4, Wn=lowpass_frequency, fs=frequency, btype='low')
    Y_predicted_filtered = np.copy(Y_predicted)
    for i in range(Y_predicted.shape[1]):
        Y_predicted_filtered[:, i] = scipy.signal.filtfilt(b_lowpass_2Hz, a_lowpass_2Hz, Y_predicted[:, i])
    assert Y_predicted.shape == Y_predicted_filtered.shape
    return Y_predicted_filtered


def best_channels_greedy_search(X_train_3D, Y_train, X_val_3D, Y_val, max_number_of_channels=None):
    best_channels_list = []
    if max_number_of_channels is None:
        max_number_of_channels = X_train_3D.shape[1]

    while len(best_channels_list) < max_number_of_channels:
        current_channels_result = {}
        for channel in set(range(X_train.shape[1])) - set(best_channels_list):
            current_best_channels =  best_channels_list + [channel]
            model = sklearn.linear_model.LinearRegression()
            model.fit(make_flat(X_train_3D[:, current_best_channels, :]), Y_train)
            Y_predicted = model.predict(make_flat(X_val_3D[:, current_best_channels, :]))
            test_corr = np.corrcoef(Y_predicted.reshape(1, -1), Y_val.reshape(1, -1))[0, 1]
            current_channels_result[channel] = test_corr
            print(f"Try add {channel} got {round(test_corr, 2)} correlation")
        print(f"Current best {best_channels_list}")
        best_channels_list.append(max(current_channels_result, key=current_channels_result.get))
    print(f"Final best {best_channels_list}")
    return best_channels_list


ALPHA_RANGE = [20, 10, 7, 5, 4, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
def get_best_alpha(X_train, Y_train, X_test, Y_test, frequency, model_class):
    max_corr = -1
    best_alpha = None
    for alpha in ALPHA_RANGE:
        model = model_class(alpha=alpha, fit_intercept=True, normalize=True)
        model.fit(X_train_new, Y_train)
        Y_predicted = model.predict(X_test_new)
        test_corr = np.corrcoef(Y_predicted_filtered.reshape(1, -1), Y_test.reshape(1, -1))[0, 1]
        if test_corr > max_corr:
            best_alpha = alpha
            max_corr = test_corr
        print(f"Alpha {alpha}: {round(test_corr, 2)} correlation")
    print(f"best_alpha: {best_alpha}")
    return best_alpha


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
