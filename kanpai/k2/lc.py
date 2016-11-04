import numpy as np
from everest import Everest
import statsmodels.api as sm
from astropy.stats import sigma_clip

import util


def everest_folded(epic, p, t0, t14, width=0.8, clip=False, bl=False):

    star = Everest(epic)
    t, f = star.time, star.flux
    idx = np.isnan(t) | np.isnan(f)
    t, f = t[~idx], f[~idx]
    tf, ff = util.fold(t, f, p, t0, t14=t14, width=width, clip=clip, bl=bl)

    return tf, ff
