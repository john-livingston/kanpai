import numpy as np
import pandas as pd

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
        top2 = util.stats.pca(X, n=2)
        aux = np.c_[bias, top2].T

    elif method == 'pca2':

        X = np.c_[pix, pix**2].T
        top2 = util.stats.pca(X, n=2)
        aux = np.c_[bias, top2].T

    elif method == 'pca-quad':

        X = pix.T
        top2 = util.stats.pca(X, n=2)
        aux = np.c_[bias, top2, top2**2].T

    else:
            raise ValueError('method must be one of: {}'.format(METHODS))

    return aux


def make_samples_h5(npz_fp, p):

    npz = np.load(npz_fp)
    df = pd.DataFrame(dict(zip(npz['pv_names'], npz['flat_chain'].T)))

    df['rhostar'] = util.transit.rhostar(p, df['a'])
    df['t14_k'] = util.transit.t14_circ(p, df['a'], df['k_k'], df['b'])
    df['t14_s'] = util.transit.t14_circ(p, df['a'], df['k_s'], df['b'])
    df['k'] = df['k_k k_s'.split()].mean(axis=1)
    df['i'] = util.transit.inclination(df['a'], df['b']) * 180 / np.pi
    df['t14'] = util.transit.t14_circ(p, df['a'], df['k'], df['b'])
    df['t23'] = util.transit.t23_circ(p, df['a'], df['k'], df['b'])
    df['tshape'] = df['t23'] / df['t14']
    df['tau'] = util.transit.tau_circ(p, df['a'], df['k'], df['b'])
    df['max_k'] = util.transit.max_k(df['tshape'])

    fp = npz_fp.replace('.npz', '-samples.h5')
    df.to_hdf(fp, key='samples')


def make_latex(h5_fp):
    pass
