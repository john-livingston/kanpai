import os
import sys
import yaml

import numpy as np
np.warnings.simplefilter('ignore')
import pandas as pd

from . import prob
from . import mod
from .. import plot
from .. import util
from .. import engines
from ..fit import Fit
from .plot import corrected_ts


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


    def summarize_mcmc(self):

        summary = {}
        summary['pv_best'] = dict(zip(self._pv_names, self._pv_best.tolist()))
        summary['logprob_best'] = float(self._lp_best)
        if len(self._gr.shape) > 1:
            gr = self._gr[-1,:]
        else:
            gr = self._gr
        summary['gelman_rubin'] = dict(zip(self._pv_names, gr.tolist()))

        percs = [15.87, 50.0, 84.13]
        pc = np.percentile(self._fc, percs, axis=0).T.tolist()
        summary['percentiles'] = dict(zip(self._pv_names, pc))

        resid_spz = self.resid(pv=self._pv_best)
        rms_spz = util.stats.rms(resid_spz)

        tstep = np.median(np.diff(self._data[:,0])) * 86400
        beta_spz = util.stats.beta(resid_spz, tstep)

        nd, npar = self._data[:,0].shape[0], len(self._pv_best)
        idx = self._pv_names.index('s')
        sigma_spz = np.median(self._fc[:,idx])
        rchisq_spz = util.stats.chisq(resid_spz, sigma_spz, nd, npar, reduced=True)
        bic_spz = util.stats.bic(self._lp_mcmc, nd, npar)

        summary['spz_rms'] = float(rms_spz)
        summary['spz_beta'] = float(beta_spz)
        summary['spz_rchisq'] = float(rchisq_spz)
        summary['spz_bic'] = float(bic_spz)

        fp = os.path.join(self._out_dir, 'mcmc-summary.yaml')
        yaml.dump(summary, open(fp, 'w'), default_flow_style=False)


    def post_mcmc(self):

        self._make_df_spz()
        fp = os.path.join(self._out_dir, 'spz.csv')
        self._data_spz.to_csv(fp, index=False)
        self._plot_corrected()


    def _make_df_spz(self):

        if self._logprob is prob.logprob_q:
            spz_model = mod.model_q
        elif self._logprob is prob.logprob_u:
            spz_model = mod.model_u
        else:
            sys.exit('logprob not one of: logprob_u, logprob_q')

        t, f = self._data.T
        self._data_spz = pd.DataFrame(dict(t=t, f=f))

        args_mod = self._args[:-1]
        mod_full = spz_model(self._pv_best, *args_mod)
        mod_transit = spz_model(self._pv_best, *args_mod, ret_ma=True)
        mod_sys = spz_model(self._pv_best, *args_mod, ret_sys=True)
        resid = f - mod_full
        fcor = f - mod_sys

        self._data_spz['f_cor'] = fcor
        self._data_spz['resid'] = resid
        self._data_spz['mod_full'] = mod_full
        self._data_spz['mod_transit'] = mod_transit
        self._data_spz['mod_sys'] = mod_sys


    def _plot_corrected(self, pv=None):

        if pv is None:
            pv = self._pv_best

        t, f = self._data_spz['t f'.split()].values.T
        mod_transit = self._data_spz['mod_transit'].values
        mod_full = self._data_spz['mod_full'].values
        f_cor = self._data_spz['f_cor'].values
        resid = self._data_spz['resid'].values
        fp = os.path.join(self._out_dir, 'fit-best.png')
        corrected_ts(t, f, f_cor, mod_full, mod_transit, resid, fp)
