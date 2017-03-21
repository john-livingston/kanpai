import os
import sys
import yaml

import numpy as np
np.warnings.simplefilter('ignore')

from . import prob
from .. import plot
from .. import util
from .. import engines
from ..fit import Fit


class FitK2(Fit):

    def __init__(self, t, f, k=None, tc=0, t14=0.2, p=20, b=0, out_dir=None, logprob=prob.logprob_q, sc=False):

        self._data = np.c_[t,f]
        if k is None:
            k = np.sqrt(1-f.min())
        self._k = k
        self._tc = tc
        self._t14 = t14
        self._p = p
        self._b = b
        self._out_dir = out_dir
        self._logprob = logprob
        self._sc = sc
        self._ld_prior = None
        super(self.__class__, self).__init__()

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
        k0 = 0
        t, f = self._data.T
        ls = np.log(f.std())
        a = util.transit.scaled_a(p, t14, k, np.pi/2)
        pv = [k,tc,a,b,q1,q2,ls,k0]
        return np.array(pv)

    @property
    def _args(self):
        t, f = self._data.T
        p = self._p
        ldp = self._ld_prior
        sc = self._sc
        return t, f, p, ldp, sc


class FitK2Tc(Fit):

    def __init__(self, t, f, p, ps, out_dir=None, sc=False):

        self._data = np.c_[t,f]
        self._p = p
        self._ps = ps
        self._out_dir = out_dir
        self._logprob = prob.logprob_q_tc
        self._sc = sc
        super(self.__class__, self).__init__()

    @property
    def _ini(self):
        tc = self._data[:,0].mean()
        ls = np.log(self._data[:,1].std())
        k0 = 0
        pv = [tc,ls,k0]
        return np.array(pv)

    @property
    def _args(self):
        t, f = self._data.T
        p = self._p
        ps = self._ps
        sc = self._sc
        return t, f, p, ps, sc
