from __future__ import absolute_import
from __future__ import print_function
import os
import functools
import numpy as np
import pandas as pd
from six.moves import map
from six.moves import range
from six.moves import zip
try:
    from photutils.centroids import centroid_com, centroid_2dg
except:
    from photutils.morphology import centroid_com, centroid_2dg

import sxp

from . import plot
from .. import util


def load_spz(data_dir, aor, radius=2.4, geom='3x3', binsize=60, rescale=None, make_plots=True, out_dir=None):

    if make_plots:
        assert out_dir is not None

    print("\nLoading Spitzer data: {}".format(aor))
    print("Radius [pixels]= {}".format(radius))
    print("Bin size [seconds] = {}".format(binsize))
    print("Geometry = {}".format(geom))

    if geom == '3x3':
        npix = 9
    elif geom == '5x5':
        npix = 25
    else:
        sys.exit('Geometry must be one of: 3x3, 5x5')

    fp = os.path.join(data_dir, aor+'_phot.pkl')
    df = sxp.util.df_from_pickle(fp, radius, pix=True, geom=geom)

    # rescale errorbars if desired
    if rescale is not None:
        df['s'] *= rescale

    # plot
    if make_plots:
        fp = os.path.join(out_dir, 'spz_raw.png')
        plot.errorbar(df.t, df.f, df.s, fp, alpha=0.5)

    # extract time series and bin
    keys = ['p{}'.format(i) for i in range(npix)]
    pix = df[keys].values
    t, f, s = df.t, df.f, df.s
    if binsize > 0:
        timestep = np.median(np.diff(t)) * 24 * 3600
        bs = int(round(binsize/timestep))
        binned = functools.partial(util.ts.binned, binsize=bs)
        tb, fb, ub, pixb = list(map(binned, [t, f, s, pix]))
        ub /= np.sqrt(bs)
        t, f, s, pix = tb, fb, ub, pixb
        if make_plots:
            fp = os.path.join(out_dir, 'spz_binned.png')
            plot.errorbar(tb, fb, ub, fp)
    d_sp = {k:v for k,v in zip('t f s'.split(), [t, f, s])}
    df = pd.DataFrame(d_sp)

    side = int(np.sqrt(npix))
    cube = pix.reshape(-1,side,side)
    cubestacked = np.median(cube, axis=0)
    if make_plots:
        fp = os.path.join(out_dir, 'spz_pix.png')
        plot.pixels(cubestacked, fp)

    # compute and plot centroids
    cx, cy = centroid_2dg(cubestacked)
    print("Cube centroids: {}, {}".format(cx, cy))
    cx, cy = list(map(int, list(map(round, [cx, cy]))))

    xy = np.array([centroid_com(i) for i in cube])
    x, y = xy.T
    df['x'] = x
    df['y'] = y
    if make_plots:
        fp = os.path.join(out_dir, 'spz_cen.png')
        plot.centroids(t, x, y, fp)

    return df, pix
