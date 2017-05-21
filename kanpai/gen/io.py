from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd

from .. import util


def load_fukui(fp, binning=None, trim=None):

    """
    If supplied, binning should be the desired bin size in seconds.
    If supplied, trim should be a tuple of desired data range in BJD_TDB.
    """

    names = 'BJD(TDB)-2450000 flux err airmass sky dx dy fwhm peak baserun'.split()
    df = pd.read_table(fp, names=names, skiprows=1, delimiter=' ')
    df = df.drop('baserun', axis=1)
    df['t_bjd_tdb'] = df['BJD(TDB)-2450000'] + 2450000

    if binning is not None:

        t = df['t_bjd_tdb']
        binning /= 86400.
        binsize = int(round(binning / np.diff(t).mean()))

        bd = util.ts.binned(df, binsize=binsize)
        df = pd.DataFrame(dict(zip(df.columns, bd.T)))

        df['err'] = df['err'] / np.sqrt(binsize)

    if trim is not None:

        t_range = df.t.max() - df.t.min()
        idx = (df.t > trim[0]) & (df.t < trim[1])
        print("Trimming {} data points outside of desired window".format(idx.sum()))
        df = df[idx]

    return df
