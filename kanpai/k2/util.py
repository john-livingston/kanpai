from __future__ import absolute_import
import numpy as np
import pandas as pd

from .. import util
from six.moves import zip


def make_samples_h5(npz_fp, p):

    npz = np.load(npz_fp)
    df = pd.DataFrame(dict(list(zip(npz['pv_names'], npz['flat_chain'].T))))

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
