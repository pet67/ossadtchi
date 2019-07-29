import scipy
import scipy.io

import numpy as np


def load_ecog_pn_mat(filepath):
    data = scipy.io.loadmat(filepath)
    X = data["ECOG"]
    Y = data["PN"]
    return X, Y


def load_finger_ost_30_mat(filepath):
    data = scipy.io.loadmat(filepath)
    X = np.transpose(np.copy(data['X'][0][0][0][:16,16000:130000]))
    Y = np.transpose(np.copy(data['X'][0][0][0][17:20,16000:130000]))
    return X, Y


import h5py

def interpolateRow(y):
    nans = np.isnan(y)
    if sum(~nans) == 0:
        y = np.nan
    else:
        y[nans] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], y[~nans])
    return y

def interpolatePN(y, empty_fill_val=0):
    np.apply_along_axis(interpolateRow,0,y)
    return y

def h5_data_loader(path):
    myo_range=range(0, 64 - 7)
    fingersrange=[15, 41, 65, 89, 113, 135, 155, 185, 209, 233]
    fingersrange=[f + 69 - 7 for f in fingersrange]
    left_cut=10003
    with h5py.File(path,'r+') as f1:
        raw_data = np.array(f1['protocol1']['raw_data'])
        pnfingers=np.copy(raw_data[:,fingersrange])
        myo=np.copy(raw_data[:,myo_range])
    pnfingers=interpolatePN(pnfingers)

    return myo, pnfingers
