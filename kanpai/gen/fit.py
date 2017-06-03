from __future__ import absolute_import
import os
import sys
import yaml

import numpy as np
from six.moves import zip
np.warnings.simplefilter('ignore')
import pandas as pd

from . import prob
from . import mod
from .. import plot
from .. import util
from .. import engines
from ..fit import Fit


class FitGen(Fit):

    def __init__(self, t, f, k=None, tc=None, t14=0.2, p=20, b=0, q1=0.1, q2=0.1, aux=None, out_dir=None, logprob=prob.logprob_q):

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
        self._q1 = q1
        self._q2 = q2
        if aux is None:
            n = self._data.shape[0]
            bias = np.repeat(1, n)
            aux = bias.reshape(1, n)
        self._aux = aux
        self._logprob = logprob
        self._out_dir = out_dir
        self._ld_prior = None
        self._map = None
        self._mcmc = None

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
        q1 = self._q1
        q2 = self._q2
        k1 = 0
        t, f = self._data.T
        idx = (t < tc - t14/2.) | (tc + t14/2. < t)
        ls = np.log(f.std())
        a = util.transit.scaled_a(p, t14, k)
        pv = [k,tc,a,b,q1,q2,ls,k1]
        if self._logprob is prob.logprob_gp:
            pv += [0] * 2
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


    def summarize_mcmc(self):

        summary = {}
        summary['pv_best'] = dict(list(zip(self._pv_names, self._pv_best.tolist())))
        summary['logprob_best'] = float(self._lp_best)
        if len(self._gr.shape) > 1:
            gr = self._gr[-1,:]
        else:
            gr = self._gr
        summary['gelman_rubin'] = dict(list(zip(self._pv_names, gr.tolist())))
        summary['acceptance_fraction'] = float(np.median(self._af))

        percs = [15.87, 50.0, 84.13]
        pc = np.percentile(self._fc, percs, axis=0).T.tolist()
        summary['percentiles'] = dict(list(zip(self._pv_names, pc)))

        _resid = self.resid(pv=self._pv_best)
        _rms = util.stats.rms(_resid)

        tstep = np.median(np.diff(self._data[:,0])) * 86400
        _beta = util.stats.beta(_resid, tstep)

        nd, npar = self._data[:,0].shape[0], len(self._pv_best)
        idx = self._pv_names.index('ls')
        _sigma = np.exp(np.median(self._fc[:,idx]))
        _rchisq = util.stats.chisq(_resid, _sigma, nd, npar, reduced=True)
        _bic = util.stats.bic(self._lp_mcmc, nd, npar)

        summary['rms'] = float(_rms)
        summary['beta'] = float(_beta)
        summary['rchisq'] = float(_rchisq)
        summary['bic'] = float(_bic)

        fp = os.path.join(self._out_dir, 'mcmc-summary.yaml')
        yaml.dump(summary, open(fp, 'w'), default_flow_style=False)


    def post_mcmc(self):

        self._make_df()
        fp = os.path.join(self._out_dir, 'output.csv')
        self._df.to_csv(fp, index=False)
        self._plot_corrected()


    def _make_df(self):

        if self._logprob is prob.logprob_q:
            _model = mod.model_q
        elif self._logprob is prob.logprob_gp:
            _model = mod.model_gp
        else:
            sys.exit('logprob not one of: logprob_q, logprob_gp')

        t, f = self._data.T
        self._df = pd.DataFrame(dict(t=t, f=f))

        args_mod = self._args[:-1]
        mod_full = _model(self._pv_best, *args_mod)
        mod_transit = _model(self._pv_best, *args_mod, ret_ma=True)
        mod_sys = _model(self._pv_best, *args_mod, ret_sys=True)
        resid = f - mod_full
        fcor = f - mod_sys

        self._df['f_cor'] = fcor
        self._df['resid'] = resid
        self._df['mod_full'] = mod_full
        self._df['mod_transit'] = mod_transit
        self._df['mod_sys'] = mod_sys


    def _plot_corrected(self, pv=None):

        if pv is None:
            pv = self._pv_best

        t, f = self._df['t f'.split()].values.T
        mod_transit = self._df['mod_transit'].values
        mod_full = self._df['mod_full'].values
        f_cor = self._df['f_cor'].values
        resid = self._df['resid'].values
        fp = os.path.join(self._out_dir, 'fit-best.png')
        plot.corrected_ts(t, f, f_cor, mod_full, mod_transit, resid, fp)
