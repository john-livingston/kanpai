import numpy as np
from everest import Everest
import k2plr as kplr

from .. import util
from fit import Fit


PIPELINES = 'everest k2sff k2sc'.split()
K2_TIME_OFFSET = 2454833


def unfolded(epic, p, t0, t14, pipeline='everest'):

    epic = int(epic)
    if pipeline == 'everest':
        star = Everest(epic)
        star.set_mask(transits = [(p, t0, t14)])
        t, f = star.time, star.flux
    elif pipeline == 'k2sff':
        star = kplr.K2SFF(epic)
        t, f = star.time, star.fcor
    elif pipeline == 'k2sc':
        star = kplr.K2SC(epic)
        t, f = star.time, star.pdcflux
    else:
        raise ValueError('Pipeline must be one of: {}'.format(PIPELINES))

    t, f = map(np.array, (t, f))
    t += K2_TIME_OFFSET

    bad = (t == K2_TIME_OFFSET) | np.isnan(f)
    t, f = t[~bad], f[~bad]

    idx = np.argsort(t)
    t, f = t[idx], f[idx]

    f /= np.median(f)
    return t, f


def folded(epic, p, t0, t14, pipeline='everest',
    width=0.8, clip=False, bl=False, skip=None, refine=False):

    t, f = unfolded(epic, pipeline=pipeline)

    idx = np.isnan(t) | np.isnan(f)
    t, f = t[~idx], f[~idx]
    tf, ff = util.lc.fold(t, f, p, t0, t14=t14,
        width=width, clip=clip, bl=bl, skip=skip)

    if refine:
        fit = Fit(tf, ff, t14=t14, p=p)
        fit.run_map()
        t14 = fit.t14()
        tf, ff = util.lc.fold(t, f, p, t0, t14=t14,
            width=width, clip=clip, bl=bl, skip=skip)
        print "Refined transit duration: {} [days]".format(t14)

    return tf, ff
