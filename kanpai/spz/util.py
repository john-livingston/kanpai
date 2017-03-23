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

    df['i'] = util.transit.inclination(df['a'], df['b']) * 180 / np.pi
    df['rhostar'] = util.transit.rhostar(p, df['a'])
    df['t14'] = util.transit.t14_circ(p, df['a'], df['k'], df['b'])
    df['t23'] = util.transit.t23_circ(p, df['a'], df['k'], df['b'])
    df['tau'] = util.transit.tau_circ(p, df['a'], df['k'], df['b'])
    df['tshape'] = df['t23'] / df['t14']
    df['max_k'] = util.transit.max_k(df['tshape'])

    cols = 'a b i k ls rhostar t14 t23 tau tshape max_k'.split()
    fp = npz_fp.replace('.npz', '-samples.h5')
    df[cols].to_hdf(fp, key='samples')

    qt = df[cols].quantile([0.1587, 0.5, 0.8413]).transpose()
    fp = npz_fp.replace('.npz', '-quantiles.txt')
    with open(fp, 'w') as w:
        w.write(qt.to_string() + '\n')

    fmt_str = '&${0:.4f}^{{+{1:.4f}}}_{{-{2:.4f}}}$'
    formatter = lambda x: fmt_str.format(x.values[1], x.values[2]-x.values[1], x.values[1]-x.values[0])
    tex = qt.apply(formatter, axis=1)
    fp = npz_fp.replace('.npz', '-quantiles-latex.txt')
    with open(fp, 'w') as w:
        w.write(tex.to_string() + '\n')
