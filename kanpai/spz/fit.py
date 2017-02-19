import os
import sys
import yaml
import pickle
import functools

import matplotlib.pyplot as pl
import numpy as np
np.warnings.simplefilter('ignore')
import pandas as pd
import scipy.optimize as op
from scipy import stats
from emcee import MHSampler, EnsembleSampler, PTSampler
from emcee.utils import sample_ball
import corner
from tqdm import tqdm

from like import logprob
from .. import plot
from .. import util
from .. import engines


class Fit(object):

    def __init__(self, t, f, k=None, tc=None, t14=0.2, p=20, b=0, aux=None, out_dir=None):

        self._data = np.c_[t,f]
        if k is None:
            k = np.sqrt(1-f.min())
        self._k = k
        if tc is None:
            tc = self._data[:,0].mean()
        self._tc = tc
        self._t14 = t14
        self._p = p
        self._b = b
        if aux is None:
            n = self._data.shape[0]
            bias = np.repeat(1, n)
            aux = bias.reshape(1, n)
        self._aux = aux
        self._logprob = logprob
        self._out_dir = out_dir
        self._ld_prior = None

    @property
    def _ini(self):
        k = self._k
        if k is None:
            f = self._data.T[1]
            k = np.sqrt(np.median(f)-f.min())
        tc = self._tc
        p = self._p
        t14 = self._t14
        b = self._b
        q1 = 0.5
        q2 = 0.5
        k1 = 0
        t, f = self._data.T
        idx = (t < tc - t14/2.) | (tc + t14/2. < t)
        sig = f[idx].std()
        a = util.transit.scaled_a(p, t14, k, np.pi/2)
        return k,tc,a,b,q1,q2,sig,k1


    @property
    def _args(self):
        t, f = self._data.T
        p = self._p
        aux = self._aux
        ldp = self._ld_prior
        return t, f, p, aux, ldp

    def set_ld_prior(self, ldp):
        self._ld_prior = ldp

    def run_map(self, methods=('nelder-mead', 'powell'), make_plots=True):

        """
        Run a maximum a posteriori (MAP) fit using one or more methods.
        Defaults to Nelder-Mead and Powell.
        """

        self._map = engines.MAP(self._logprob, self._ini, self._args, methods=methods)
        self._map.run()
        self._pv_map, self._lp_map, self._max_apo_alg = self._map.results
        self._pv_best = self._pv_map

        if make_plots:
            t, f, p, aux, ldp = self._args
            m = self._logprob(self._pv_map, *self._args, ret_mod=True)
            fp = os.path.join(self._out_dir, 'map-bestfit.png')
            plot.simple_ts(t, f, model=m, fp=fp)


    def model(self, t=None):
        p = self._p
        if t is None:
            t = self._data[:,0]
        f = np.ones_like(t)
        m = self._logprob(self._pv_best, *self._args, ret_mod=True)
        return m

    @property
    def resid(self):
        t, f = self._data.T
        return f - self.model()

    def plot(self, fp=None, nmodel=None, **kwargs):
        t, f = self._data.T
        title = "Std. dev. of residuals: {}".format(np.std(self.resid))
        if nmodel is not None:
            ti = np.linspace(t.min(), t.max(), nmodel)
            m = self.model(t=ti)
            plot.simple_ts(t, f, tmodel=ti, model=m, fp=fp, title=title, **kwargs)
        else:
            plot.simple_ts(t, f, model=m, fp=fp, title=title, **kwargs)

    def t14(self, nmodel=1000):
        t = self._data.T[0]
        ti = np.linspace(t.min(), t.max(), nmodel)
        mi = self.model(ti)
        idx = mi < 1
        t14 = ti[idx][-1] - ti[idx][0]
        return t14

    @property
    def _pv_names(self):
        return self._logprob(1, 1, 1, 1, ret_pvnames=True)

    @property
    def best(self):
        return dict(zip(self._pv_names, self._pv_best))

    def run_mcmc(self, make_plots=True, **kwargs):

        ini = self._pv_map
        args = self._args
        names = self._pv_names
        t, f, p, aux, ldp = args

        eng = engines.MCMC(self._logprob, ini, args, names, outdir=self._out_dir)
        sig_idx = self._pv_names.index('s')
        eng.run(make_plots=make_plots, pos_idx=sig_idx, **kwargs)
        pv, lp, fc, gr, acor = eng.results
        if lp > self._lp_map:
            self._pv_best = pv

        if make_plots:

            m = self._logprob(pv, *self._args, ret_mod=True)
            fp = os.path.join(self._out_dir, 'mcmc-bestfit.png')
            plot.simple_ts(t, f, model=m, fp=fp)

            fp = os.path.join(self._out_dir, 'mcmc-samples.png')
            ps = [self._logprob(s, *self._args, ret_mod=True) for s in fc[np.random.randint(len(fc), size=100)]]
            plot.samples(t, f, ps, fp=fp)
