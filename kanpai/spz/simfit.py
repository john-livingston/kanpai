import os
import sys
import yaml

import numpy as np
np.warnings.simplefilter('ignore')

from . import prob
from .. import plot
from .. import util
from .. import engines
from .mod import model_q as spz_model
from .plot import corrected_ts
from .fit import FitSpz
from ..k2.fit import FitK2
from ..fit import Fit


def logprob(theta, lp_k2, lp_spz, args_k2, args_spz, aux=None, ret_pvnames=False):

    if ret_pvnames:
        pvn = 'a,b,k_s,k_k,tc_s,tc_k,q1_s,q2_s,q1_k,q2_k,s_s,s_k,k1_s,k0_k'.split(',')
        if aux is not None:
            pvn += ['c{}'.format(i) for i in range(len(aux))]
        return pvn

    theta_k2 = get_theta(theta, 'k2')
    theta_spz = get_theta(theta, 'spz')

    lp = lp_k2(theta_k2, *args_k2) + lp_spz(theta_spz, *args_spz)

    return lp


def get_theta(theta, sub):

    a,b,k_s,k_k,tc_s,tc_k,q1_s,q2_s,q1_k,q2_k,s_s,s_k,k1_s,k0_k = theta[:14]
    theta_aux = theta[14:]

    theta_k2 =  [k_k,tc_k,a,b,q1_k,q2_k,s_k,k0_k]
    theta_sp = [k_s,tc_s,a,b,q1_s,q2_s,s_s,k1_s] + theta_aux.tolist()

    if sub == 'k2':
        return theta_k2
    elif sub == 'spz':
        return theta_sp


class FitK2Spz(Fit):

    def __init__(self, setup, data_k2, data_spz, aux=None, out_dir=None):

        self._setup = setup
        self._out_dir = out_dir
        self._tr  = setup['transit']

        self._data_k2 = data_k2
        self._data_spz = data_spz

        self._k_k = self._tr['k']
        self._k_s = self._tr['k']
        self._tc_k = 0
        self._tc_s = self._data_spz['t'].mean()
        self._t14 = self._tr['t14']
        self._p = self._tr['p']
        self._b = 0

        if aux is None:
            n = self._data_spz.shape[0]
            bias = np.repeat(1, n)
            aux = bias.reshape(1, n)
        self._aux = aux

        self._fit_k2 = FitK2(*data_k2['t f'.split()].values.T, p=self._p)
        self._fit_spz = FitSpz(*data_spz['t f'.split()].values.T, p=self._p, aux=self._aux)

        self._logprob = logprob
        self._ld_prior = None


    @property
    def _ini(self):
        k_k = self._k_k
        k_s = self._k_k
        tc_k = self._tc_k
        tc_s = self._tc_s
        p = self._p
        t14 = self._t14
        b = self._b
        q1_k = 0.5
        q2_k = 0.5
        q1_s = 0.1
        q2_s = 0.2
        k0_k = 0
        k1_s = 0
        s_k = self._data_k2['f'].std()
        s_s = self._data_spz['f'].std()
        a = util.transit.scaled_a(p, t14, k_k, np.pi/2)
        pv = [a,b,k_s,k_k,tc_s,tc_k,q1_s,q2_s,q1_k,q2_k,s_s,s_k,k1_s,k0_k]
        if self._aux is not None:
            pv += [0] * self._aux.shape[0]
        return np.array(pv)


    @property
    def _args(self):
        lp_k2 = self._fit_k2._logprob
        lp_spz = self._fit_spz._logprob
        args_k2 = self._fit_k2._args
        args_spz = self._fit_spz._args
        aux = self._aux
        return lp_k2, lp_spz, args_k2, args_spz, aux


    def post_map(self):

        pv = get_theta(self._pv_map, 'k2')
        fp = os.path.join(self._out_dir, 'map-k2.png')
        self._fit_k2.plot(fp=fp, nmodel=1000, pv=pv)

        pv = get_theta(self._pv_map, 'spz')
        fp = os.path.join(self._out_dir, 'map-spz.png')
        self._fit_spz.plot(fp=fp, pv=pv)


    def post_mcmc(self):

        self._make_df_spz()
        fp = os.path.join(self._out_dir, 'spz.csv')
        self._data_spz.to_csv(fp, index=False)

        self._plot_corrected()

        fc = self._mcmc._fc
        names = self._mcmc._names

        fp = os.path.join(self._out_dir, 'mcmc-corner.png')
        plot.corner(fc, names, fp=fp)

        t, f = self._data_k2['t f'.split()].values.T
        ps = [self._fit_k2.model(pv=get_theta(s, 'k2')) for s in fc[np.random.randint(len(fc), size=100)]]
        fp = os.path.join(self._out_dir, 'mcmc-samples-k2.png')
        plot.samples(t, f, ps, fp=fp)

        t, f = self._data_spz['t f'.split()].values.T
        ps = [self._fit_spz.model(pv=get_theta(s, 'spz')) for s in fc[np.random.randint(len(fc), size=100)]]
        fp = os.path.join(self._out_dir, 'mcmc-samples-spz.png')
        plot.samples(t, f, ps, fp=fp)


    def _make_df_spz(self):

        pv_best = get_theta(self._pv_best, 'spz')
        args_mod = self._fit_spz._args[:-1]
        mod_full = spz_model(pv_best, *args_mod)
        mod_transit = spz_model(pv_best, *args_mod, ret_ma=True)
        mod_sys = spz_model(pv_best, *args_mod, ret_sys=True)
        resid = self._data_spz['f'] - mod_full
        fcor = self._data_spz['f'] - mod_sys
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
