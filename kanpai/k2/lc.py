import numpy as np
from everest import Everest
import k2plr as kplr
import statsmodels.api as sm
from astropy.stats import sigma_clip

import util

PIPELINES = 'everest k2sff k2sc'.split()


def folded(epic, p, t0, t14, pipeline='everest',
    width=0.8, clip=False, bl=False, skip=None):

    if pipeline == 'everest':
        star = Everest(epic)
        t, f = star.time, star.flux
    elif pipeline == 'k2sff':
        star = kplr.K2SFF(epic)
        t, f = star.time, star.fcor
    elif pipeline == 'k2sc':
        star = kplr.K2SC(epic)
        t, f = star.time, star.pdcflux
    else:
        raise ValueError('pipeline must be one of: {}'.format(PIPELINES))

    idx = np.isnan(t) | np.isnan(f)
    t, f = t[~idx], f[~idx]
    tf, ff = util.fold(t, f, p, t0, t14=t14, 
        width=width, clip=clip, bl=bl, skip=skip)

    return tf, ff
