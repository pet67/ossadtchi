import scipy
import scipy.io


def load_ecog_pn_mat(filepath):
    data = scipy.io.loadmat(filepath)
    X = data["ECOG"]
    Y = data["PN"]
    return X, Y
