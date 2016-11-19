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
        self._t0 = t0 - K2_TIME_OFFSET
        self._t14 = t14
        self._pipeline = pipeline
        self._width = width
        self._clip = clip
        self._bl = bl
        self._skip = skip
        self._get_lc()


    def _get_lc(self):

        epic = self._epic
        pipeline = self._pipeline
        print("Retrieving {} light curve...".format(pipeline))

        if pipeline == 'everest':

            star = Everest(epic)
            star.set_mask(transits = [(self._p, self._t0, self._t14)])
            t, f = star.time, star.flux

        elif pipeline == 'k2sff':

            star = kplr.K2SFF(epic)
            t, f = star.time, star.fcor

        elif pipeline == 'k2sc':

            star = kplr.K2SC(epic)
            t, f = star.time, star.pdcflux

        else:

            raise ValueError('Pipeline must be one of: {}'.format(PIPELINES))

        idx = np.isnan(t) | np.isnan(f)
        self._t, self._f = t[~idx], f[~idx]


    def run(self, refine=True):

        # setup
        t, f = self._t, self._f
        p, t0, t14 = self._p, self._t0, self._t14
        w, bl, s = self._width, self._bl, self._skip

        # initial fold
        tf, ff = util.fold(t, f, p, t0, t14=t14,
            width=w, bl=bl, skip=s)

        # outlier rejection
        idx = util.outliers(ff, sl=5, su=3, iterative=True)
        tf, ff = tf[~idx], ff[~idx]
        print("1st sigma clip: {}".format(idx.sum()))

        if not refine:
            self._tf, self._ff = tf, ff
            return

        # initial fit
        fit = Fit(tf, ff, t14=t14, p=p)
        fit.max_apo()
        t14 = fit.t14()

        # fold with refined t14 to ensure proper baseline removal
        tf, ff = util.fold(t, f, p, t0, t14=t14,
            width=w, bl=bl, skip=s)

        # outlier rejection
        idx = util.outliers(ff, sl=5, su=3, iterative=True)
        tf, ff = tf[~idx], ff[~idx]
        print("2nd sigma clip: {}".format(idx.sum()))

        # second fit to correct for any offset in initial T0
        fit = Fit(tf, ff, t14=t14, p=p)
        fit.max_apo()
        t14 = fit.t14()
        par = fit.final()
        t0 -= par['tc']
        print "Refined T0 [BJD]: {}".format(t0 + K2_TIME_OFFSET)

        # fold with refined T0
        tf, ff = util.fold(t, f, p, t0, t14=t14,
            width=w, bl=bl, skip=s)

        # identify final outliers by sigma clipping residuals
        fit = Fit(tf, ff, t14=t14, p=p)
        fit.max_apo()
        sl, su = self._clip
        idx = util.outliers(fit.resid, sl=sl, su=su)
        tf, ff = tf[~idx], ff[~idx]
        print("3rd sigma clip: {}".format(idx.sum()))

        # final fit to cleaned light curve
        fit = Fit(tf, ff, t14=t14,
            p=p, k=par['k'], i=par['i'], u=par['u'], k0=par['k0'])
        fit.max_apo()
        t14 = fit.t14()
        par = fit.final()
        fit_a = util.scaled_a(p, t14, par['k'], par['i'])

        print "Transit duration (t14) [days]: {}".format(t14)
        print "Scaled semi-major axis (a): {}".format(fit_a)
        print "Radius ratio (k): {}".format(par['k'])
        print "Inclination (i) [degrees]: {}".format(par['i'] * 180./np.pi)
        print "Linear limb-darkening coefficient (u): {}".format(par['u'])
        print "Baseline offset (k0): {}".format(par['k0'])
        print "Sigma: {}".format(par['sig'])
        print "Residual RMS: {}".format(util.rms(fit.resid))

        self._fit = fit
        self._tf, self._ff = tf, ff


    @property
    def results(self):

        return self._tf, self._ff


    def plot_fit(self, fp):

        self._fit.plot(fp, lw=5, ms=10, nmodel=1000)
