import numpy as np
import pandas as pd

from .. import util


def load_k2(k2_folded_fp, binning=None):

    try:

        print "\nLoading K2 data from file: {}".format(k2_folded_fp)
        ncols = len(open(k2_folded_fp).readline().split(','))
        if ncols == 3:
            df = pd.read_csv(k2_folded_fp, names='t f s'.split())
        else:
            df = pd.read_csv(k2_folded_fp, names='t f'.split())
            df['s'] = np.repeat(df['f'].std(), df.shape[0])

    except:

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

    return df
