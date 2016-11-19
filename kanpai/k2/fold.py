import numpy as np
from everest import Everest
import k2plr as kplr
from astropy.stats import sigma_clip

import util
from fit import Fit


K2_TIME_OFFSET = 2454833


class Fold(object):

    def __init__(self, epic, p, t0, t14=0.2, pipeline='everest',
        width=0.8, clip=[3,3], bl=True, skip=None):

        self._epic = int(epic)
        self._p = p
        self._t0 = t0
        self._t14 = t14
        self._pipeline = pipeline
        self._width = width
        self._clip = clip
        self._bl = bl
        self._skip = skip
        self._get_lc()


    def _get_lc(self):

        print("Retrieving light curve")
        if self._pipeline == 'everest':

            star = Everest(self._epic)
            star.set_mask(transits = [(p, t0, t14)])
            t, f = star.time, star.flux

        elif self._pipeline == 'k2sff':

            star = kplr.K2SFF(self._epic)
            t, f = star.time, star.fcor

        elif self._pipeline == 'k2sc':

            star = kplr.K2SC(self._epic)
            t, f = star.time, star.pdcflux

        else:

            raise ValueError('Pipeline must be one of: {}'.format(PIPELINES))

        idx = np.isnan(t) | np.isnan(f)
        self._t, self._f = t[~idx], f[~idx]


    def run(self):

        t, f = self._t, self._f
        p, t0, t14 = self._p, self._t0, self._t14
        w, bl, s = self._width, self._bl, self._skip

        tf, ff = util.fold(t, f, p, t0, t14=t14,
            width=w, bl=bl, skip=s)

        self._fit = Fit(tf, ff, t14=t14, p=p)
        self._fit.max_apo()
        t14 = self._fit.t14()

        tf, ff = util.fold(t, f, p, t0, t14=t14,
            width=w, bl=bl, skip=s)

        self._fit = Fit(tf, ff, t14=t14, p=p)
        self._fit.max_apo()
        t14 = self._fit.t14()
        print "refined transit duration: {} [days]".format(t14)

        idx = self._outliers(self._fit.resid)
        self._tf, self._ff = tf[~idx], ff[~idx]
        self._fit = Fit(self._tf, self._ff, t14=t14, p=p)
        self._fit.max_apo()
        print("Sigma-clipped {} outliers".format(idx.sum()))


    def _outliers(self, resid, iterative=True):

        sl, su = self._clip

        if iterative:
            clip = sigma_clip(resid, sigma_upper=su, sigma_lower=sl)
            idx = clip.mask
        else:
            mu, sig = np.median(resid), np.std(resid)
            idx = (resid > mu + su * sig) | (resid < mu - sl * sig)

        return idx


    @property
    def results(self):

        return self._tf, self._ff


    def plot_fit(self, fp):

        self._fit.plot(fp, lw=5, ms=10, nmodel=1000)
