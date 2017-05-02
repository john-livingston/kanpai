from __future__ import absolute_import
import numpy as np
import pandas as pd

from .. import util
from six.moves import zip


def make_quantiles_table(npz_fp, p, save_h5=False):

    npz = np.load(npz_fp)
    df = pd.DataFrame(dict(list(zip(npz['pv_names'], npz['flat_chain'].T))))

    df['i'] = util.transit.inclination(df['a'], df['b']) * 180 / np.pi
    df['k'] = df['k_k k_s'.split()].mean(axis=1)
    df['rhostar'] = util.transit.rhostar(p, df['a'])
    df['t14'] = util.transit.t14_circ(p, df['a'], df['k'], df['b'])
    df['t23'] = util.transit.t23_circ(p, df['a'], df['k'], df['b'])
    df['tau'] = util.transit.tau_circ(p, df['a'], df['k'], df['b'])
    df['tshape'] = df['t23'] / df['t14']
    df['max_k'] = util.transit.max_k(df['tshape'])

    if save_h5:
        cols = 'a b i k k_k k_s ls_k ls_s tc_s rhostar t14 t23 tau tshape max_k'.split()
        fp = npz_fp.replace('.npz', '-samples.h5')
        df[cols].to_hdf(fp, key='samples')

    qt = df[cols].quantile([0.1587, 0.5, 0.8413]).transpose()
    fp = npz_fp.replace('.npz', '-quantiles.txt')
    with open(fp, 'w') as w:
        w.write(qt.to_string() + '\n')

    df['$a/R_{\star}$'] = df['a']
    df['$b$'] = df['b']
    df['$i$'] = df['i']
    df['$R_p/R_{\star}$'] = df['k']
    df['$R_{p,S}/R_{\star}$'] = df['k_s']
    df['$R_{p,K}/R_{\star}$'] = df['k_k']
    df['log($\sigma_S$)'] = df['ls_s']
    df['log($\sigma_K$)'] = df['ls_k']
    df['$T_{c,S}$'] = df['tc_s'] - 2454833
    df['$\rho_{\star}$'] = df['rhostar']
    df['$T_{14}$'] = df['t14']
    df['$T_{23}$'] = df['t23']
    df['$\tau$'] = df['tau']
    df['$\eta$'] = df['tshape']
    df['$R_{p,max}/R_{\star}$'] = df['max_k']

    cols = ['$a/R_{\star}$', '$R_{p,S}/R_{\star}$', '$R_{p,K}/R_{\star}$',
        '$R_p/R_{\star}$', '$b$', '$i$',
        'log($\sigma_S$)', 'log($\sigma_K$)', '$T_{c,S}$',
        '$\rho_{\star}$', '$T_{14}$', '$T_{23}$',
        '$\tau$', '$\eta$', '$R_{p,max}/R_{\star}$']

    qt = df[cols].quantile([0.1587, 0.5, 0.8413]).transpose()
    fmt_str = '&${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'
    formatter = lambda x: fmt_str.format(x.values[1], x.values[2]-x.values[1], x.values[1]-x.values[0])
    tex = qt.apply(formatter, axis=1)
    fp = npz_fp.replace('.npz', '-latex.txt')
    with open(fp, 'w') as w:
        w.write(tex.to_string() + '\n')


def agg_latex(epics, latexfilepaths, fp):
    dfs = [pd.read_fwf(qfp, names=[epic]) for epic,qfp in zip(epics,latexfilepaths)]
    df = pd.concat(dfs, axis=1)
    units = '--- --- deg. --- --- --- --- --- BKJD $g/cm^3$ day day day --- ---'.split()
    df['Unit'] = ['&'+i for i in units]
    cols = ['Unit'] + epics
    with open(fp, 'w') as w:
        for line in df[cols].to_string().split('\n'):
            w.write(line + ' \\\\\n')
    with open('transpose-'+fp, 'w') as w:
        for line in df[cols].transpose().to_string().split('\n'):
            w.write(line + ' \\\\\n')
