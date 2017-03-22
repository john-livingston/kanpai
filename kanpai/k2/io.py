import numpy as np
import pandas as pd

from .. import util


def load_k2(k2_folded_fp, binning=None, trim=None):

    try:

        print "\nLoading K2 data from file: {}".format(k2_folded_fp)
        ncols = len(open(k2_folded_fp).readline().split(','))
        if ncols == 3:
            df = pd.read_csv(k2_folded_fp, names='t f s'.split())
        else:
            df = pd.read_csv(k2_folded_fp, names='t f'.split())
            df['s'] = np.repeat(df['f'].std(), df.shape[0])

    except:
        print k2_folded_fp
        raise ValueError('Invalid K2 light curve file format')

    if binning is not None:

        binning /= 86400.

        t, f, s = df.values.T

        binsize = int(round(binning / np.diff(t).mean()))
        tb = util.ts.binned(t, binsize)
        fb = util.ts.binned(f, binsize)
        sb = util.ts.binned(s, binsize)
        sb /= np.sqrt(binsize)

        df = pd.DataFrame(dict(t=tb, f=fb, s=sb))

    if trim is not None:

        t_range = df.t.max() - df.t.min()
        if trim < t_range:
            # assume the input light curve is centered (T=0 at mid-transit)
            idx = (df.t > -trim/2.) & (df.t < trim/2.)
            print "Trimming {} data points outside of desired window".format(idx.sum())
            return df[idx]

    return df
