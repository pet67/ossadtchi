import h5py
import numpy as np
import scipy
import scipy.io


# Next two global variables for h5_data_loaders
FINGERS_RANGE = [15, 41, 65, 89, 113, 135, 155, 185, 209, 233]
FINGERS_RANGE_NAMES = [
    '18_Xrot', '22_Zrot', '26_Zrot', '30_Zrot', '34_Zrot',
    '41_Xrot', '44_Zrot', '49_Zrot', '53_Zrot', '57_Zrot'
]


def load_ecog_pn_mat(filepath):
    data = scipy.io.loadmat(filepath)
    X = data["ECOG"]
    Y = data["PN"]
    return X, Y


def load_finger_ost_30_mat(filepath):
    # make LEFT_CUT and RIGHT_CUT is recommended way to use finger_ost dataset
    LEFT_CUT = 16000
    RIGHT_CUT = 130000
    data = scipy.io.loadmat(filepath)
    X = np.transpose(np.copy(data['X'][0][0][0][:16, LEFT_CUT:RIGHT_CUT]))
    Y = np.transpose(np.copy(data['X'][0][0][0][17:20, LEFT_CUT:RIGHT_CUT]))
    return X, Y


def interpolateRow(y):
    nans = np.isnan(y)
    if sum(~nans) == 0:
        y = np.nan
    else:
        y[nans] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], y[~nans])
    return y


def interpolatePN(y, empty_fill_val=0):
    np.apply_along_axis(interpolateRow, 0, y)
    return y


def h5_data_loader_impl(path, channels_range, fingers_range, left_cut=0):
    with h5py.File(path, 'r+') as input_file:
        raw_data = np.copy(np.array(input_file['protocol1']['raw_data']))
        channels_names = np.copy(np.array(input_file['channels']))

    assert np.all(channels_names[fingers_range].to_list() == FINGERS_RANGE_NAMES), \
        f"Expected fingers {channels_names[fingers_range].to_list()}, found {FINGERS_RANGE_NAMES}"
    assert all(["pos" not in channels_name for channels_name in channels_names.to_list()]), \
        "Probably you use position as input channel"
    fingers = np.copy(raw_data[left_cut:, fingers_range])
    channels = np.copy(raw_data[left_cut:, channels_range])
    fingers = interpolatePN(fingers)
    return channels, fingers


def h5_data_loader_64_channels(path):
    left_cut = 0
    myo_range = range(0, 64)
    fingers_range = [f + 69 for f in FINGERS_RANGE]
    return h5_data_loader_impl(path, myo_range, fingers_range, left_cut)


def h5_data_loader_57_channels(path):
    left_cut = 0
    myo_range = range(0, 64 - 7)
    fingers_range = [f + 69 - 7 for f in FINGERS_RANGE]
    return h5_data_loader_impl(path, myo_range, fingers_range, left_cut)
