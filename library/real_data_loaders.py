import scipy
import scipy.io


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
