import numpy as np

from .. import util


def setup_aux(method, xy, pix):

    METHODS = 'cen pld base pca pca2 pca-quad cen-quad pld-quad'.split()

    n = xy.shape[0]
    bias = np.repeat(1, n)

    if method == 'cen':

        aux = np.c_[bias, xy].T

    elif method == 'cen-quad':

        aux = np.c_[bias, xy, xy**2].T

    elif method == 'pld':

        aux = pix.T

    elif method == 'pld-quad':

        aux = np.c_[pix, pix**2].T

    elif method == 'base':

        aux = bias.reshape(1, n)

    elif method == 'pca':

        X = pix.T
        top2 = util.pca(X, n=2)
        aux = np.c_[bias, top2].T

    elif method == 'pca2':

        X = np.c_[pix, pix**2].T
        top2 = util.pca(X, n=2)
        aux = np.c_[bias, top2].T

    elif method == 'pca-quad':

        X = pix.T
        top2 = util.pca(X, n=2)
        aux = np.c_[bias, top2, top2**2].T

    else:
            raise ValueError('method must be one of: {}'.format(METHODS))

    return aux
