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


class FitSpz(Fit):

    def __init__(self, t, f, k=None, tc=None, t14=0.2, p=20, b=0, aux=None, out_dir=None, logprob=prob.logprob_q):

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
        s = f.std()
        a = util.transit.scaled_a(p, t14, k, np.pi/2)
        pv = [k,tc,a,b,q1,q2,s,k1]
        if self._aux is not None:
            pv += [0] * self._aux.shape[0]
        return np.array(pv)


    @property
    def _args(self):
        t, f = self._data.T
        p = self._p
        aux = self._aux
        ldp = self._ld_prior
        return t, f, p, aux, ldp
