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

from like import model2, loglike2
import plot


def logprob(theta, t, f, p):

    k,tc,t14,i,u,k0,sig = theta

    if u < 0 or u > 1 or i > np.pi/2:
        return -np.inf

    ll = loglike2(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll


class Fit(object):

    def __init__(self, t, f, k=None, tc=0, t14=0.1, p=20, i=np.pi/2., u=0.5, k0=0):

        self._data = np.c_[t,f]
        self._k = k
        self._tc = tc
        self._t14 = t14
        self._p = p
        self._i = i
        self._u = u
        self._k0 = k0
        self._pv_best = None
        self._logprob_best = None
        self._logprob = logprob

    def _initial(self):
        k = self._k
        if k is None:
            f = self._data.T[1]
            k = np.sqrt(np.median(f)-f.min())
        tc = self._tc
        t14 = self._t14
        i = self._i
        # FIXME: upgrade limbdark to get linear LD coeff from LDTk
        u = self._u
        k0 = self._k0
        t, f = self._data.T
        idx = (t < tc - t14/2.) | (tc + t14/2. < t)
        sig = f[idx].std()
        return k,tc,t14,i,u,k0,sig

    def _args(self):
        t, f = self._data.T
        p = self._p
        return t, f, p

    def _map(self, method='nelder-mead'):

        initial = self._initial()
        nlp = lambda *x: -self._logprob(*x)
        args = self._args()
        res = op.minimize(nlp, initial, args=args, method=method)
        return res

    def max_apo(self, methods=('nelder-mead', 'powell')):
        results = []
        for method in methods:
            res = self._map(method=method)
            if res.success:
                print "{} negative log probability: {}".format(method, res.fun)
                results.append(res)
        if len(results) > 0:
            idx = np.argmin([r.fun for r in results])
            map_best = np.array(results)[idx]
            self._pv_best = map_best.x
            self._logprob_best = map_best.fun
        else:
            print "All methods failed to converge."
        return

    def model(self, t=None, include_offset=True):
        p = self._p
        if t is None:
            t = self._data[:,0]
        if include_offset:
            f = np.ones_like(t)
            m = loglike2(self._pv_best, t, f, p, ret_mod=True)
        else:
            m = model2(self._pv_best, t, p)
        return m

    def plot(self, fp=None, nmodel=None, **kwargs):
        t, f = self._data.T
        m = self.model()
        resid = f - m
        title = "Std. dev. of residuals: {}".format(np.std(resid))
        if nmodel is not None:
            ti = np.linspace(t.min(), t.max(), nmodel)
            m = self.model(t=ti)
            plot.simple_ts(t, f, tmodel=ti, model=m, fp=fp, title=title, **kwargs)
        else:
            plot.simple_ts(t, f, model=m, fp=fp, title=title, **kwargs)

    def t14(self, nmodel=1000):
        t = self._data.T[0]
        ti = np.linspace(t.min(), t.max(), nmodel)
        mi = self.model(ti, include_offset=False)
        idx = mi < 1
        t14 = ti[idx][-1] - ti[idx][0]
        return t14
