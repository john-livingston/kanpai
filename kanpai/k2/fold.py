import numpy as np
from everest import Everest
import k2plr as kplr
from astropy.stats import sigma_clip

from .fit import FitK2
from . import prob
from . import lc
from .. import util


class Fold(object):

    def __init__(self, epic, p, t0, t14=0.2, pipeline='everest',
        width=0.8, clip=[3,3], bl=True, skip=None, lcfp=None):

        self._epic = int(epic)
        self._p = p
        self._t0 = t0
        self._t14 = t14
        self._pipeline = pipeline
        self._width = width
        self._clip = clip
        self._bl = bl
        self._skip = skip
        self._lcfp = lcfp
        self._get_lc()


    def _get_lc(self):

        if self._lcfp is None:

            epic = self._epic
            p = self._p
            t0 = self._t0
            t14 = self._t14
            pipeline = self._pipeline
            print("Retrieving {} light curve...".format(pipeline))
            t, f = lc.unfolded(epic, p, t0, t14, pipeline=pipeline)

        else:

            t, f = np.loadtxt(self._lcfp, unpack=True)
            self._pipeline = 'user'

        idx = np.isnan(t) | np.isnan(f)
        self._t, self._f = t[~idx], f[~idx]


    def run(self, outdir, refine=True):

        # setup
        t, f = self._t, self._f
        p, t0, t14 = self._p, self._t0, self._t14
        w, bl, s = self._width, self._bl, self._skip

        # initial fold
        tf, ff = util.lc.fold(t, f, p, t0, t14=t14,
            width=w, bl=bl, skip=s)

        # outlier rejection
        idx = util.stats.outliers(ff, su=3, sl=6, iterative=True)
        tf, ff = tf[~idx], ff[~idx]
        print("1st sigma clip: {}".format(idx.sum()))

        if not refine:
            self._tf, self._ff = tf, ff
            idx = (tf < -t14/2.) | (tf > t14/2.)
            self._sig = ff[idx].std()
            return

        # initial fit
        fit = FitK2(tf, ff, t14=t14, p=p, out_dir=outdir, logprob=prob.logprob_q)
        fit.run_map(make_plots=False)
        pv = fit.best
        t14 = util.transit.t14_circ(p, pv['a'], pv['k'], pv['b'])

        # fold with refined t14 to ensure proper baseline removal
        tf, ff = util.lc.fold(t, f, p, t0, t14=t14,
            width=w, bl=bl, skip=s)

        # outlier rejection
        idx = util.stats.outliers(ff, su=3, sl=6, iterative=True)
        tf, ff = tf[~idx], ff[~idx]
        print("2nd sigma clip: {}".format(idx.sum()))

        # second fit to correct for any offset in initial T0
        fit = FitK2(tf, ff, t14=t14, p=p, out_dir=outdir, logprob=prob.logprob_q)
        fit.run_map(make_plots=False)
        pv = fit.best
        t14 = util.transit.t14_circ(p, pv['a'], pv['k'], pv['b'])
        t0 += pv['tc']
        print "Refined T0 [BJD]: {}".format(t0)

        # fold with refined T0
        tf, ff = util.lc.fold(t, f, p, t0, t14=t14,
            width=w, bl=bl, skip=s)

        # outlier rejection
        idx = util.stats.outliers(ff, su=3, sl=6, iterative=True)
        tf, ff = tf[~idx], ff[~idx]
        print("3rd sigma clip: {}".format(idx.sum()))

        # identify final outliers by sigma clipping residuals
        fit = FitK2(tf, ff, t14=t14, p=p, out_dir=outdir, logprob=prob.logprob_q)
        fit.run_map(make_plots=False)
        su, sl = self._clip
        idx = util.stats.outliers(fit.resid(), su=su, sl=sl)
        tf, ff = tf[~idx], ff[~idx]
        print("Final sigma clip ({},{}): {}".format(su, sl, idx.sum()))

        # re-normalize to median OOT flux
        idx = (tf < -t14/2.) | (tf > t14/2.)
        ff /= np.median(ff[idx])

        # final fit to cleaned light curve
        fit = FitK2(tf, ff, t14=t14, p=p, k=pv['k'], b=pv['b'],
            out_dir=outdir, logprob=prob.logprob_q)
        fit.run_map(make_plots=False)
        pv = fit.best
        k = pv['k']
        a = pv['a']
        b = pv['b']
        i = util.transit.inclination(a, b)
        t14 = util.transit.t14_circ(p, a, k, b)

        print "Transit duration (t14) [days]: {0:.4f}".format(t14)
        print "Scaled semi-major axis (a): {0:.4f}".format(a)
        print "Radius ratio (k): {0:.4f}".format(k)
        print "Impact parameter: {0:.4f}".format(b)
        print "Inclination (i) [degrees]: {0:.4f}".format(i * 180./np.pi)
        print "Limb-darkening coefficients (q1, q2): {0:.4f}, {1:.4f}".format(pv['q1'], pv['q2'])
        print "Baseline offset (k0): {0:.8f}".format(pv['k0'])
        print "Sigma: {0:.8f}".format(pv['s'])
        print "Residual RMS: {0:.8f}".format(util.stats.rms(fit.resid()))

        self._fit = fit
        self._tf, self._ff = tf, ff
        self._sig = np.std(fit.resid())


    @property
    def results(self):

        return self._tf, self._ff, np.repeat(self._sig, self._ff.size)


    def plot_fit(self, fp):

        self._fit.plot_best(fp=fp, lw=3, ms=10, nmodel=1000)
