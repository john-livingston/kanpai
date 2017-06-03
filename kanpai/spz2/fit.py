from __future__ import absolute_import
import os
import sys
import yaml
import pickle

import numpy as np
from six.moves import map
from six.moves import range
from six.moves import zip
np.warnings.simplefilter('ignore')
import pandas as pd
import sxp

from ..spz import prob
from .. import plot
from .. import util
from .. import engines
from ..spz.mod import model_q as spz_model
from ..spz.plot import corrected_ts
from .plot import i1_vs_i2
from ..spz.fit import FitSpz
from ..fit import Fit
from .util import make_quantiles_table


def logprob(theta, lp_i1, lp_i2, args_i1, args_i2, aux1=None, aux2=None, ret_pvnames=False):

    if ret_pvnames:
        pvn = 'a,b,k_i1,k_i2,tc_i1,tc_i2,q1_i1,q2_i1,q1_i2,q2_i2,ls_i1,ls_i2,k1_i1,k1_i2'.split(',')
        if aux1 is not None:
            pvn += ['c{}_i1'.format(i) for i in range(len(aux1))]
        if aux2 is not None:
            pvn += ['c{}_i2'.format(i) for i in range(len(aux2))]
        return pvn

    theta_i1 = get_theta(theta, 'i1', aux1, aux2)
    theta_i2 = get_theta(theta, 'i2', aux1, aux2)

    lp = lp_i1(theta_i1, *args_i1) + lp_i2(theta_i2, *args_i2)

    return lp


def get_theta(theta, sub, aux1, aux2):

    a,b,k_i1,k_i2,tc_i1,tc_i2,q1_i1,q2_i1,q1_i2,q2_i2,ls_i1,ls_i2,k1_i1,k1_i2 = theta[:14]
    theta_aux = theta[14:]

    naux1 = len(aux1)
    naux2 = len(aux2)

    theta_aux_i1 = theta_aux[:naux1]
    theta_aux_i2 = theta_aux[naux1:naux1+naux2]

    theta_i1 = [k_i1,tc_i1,a,b,q1_i1,q2_i1,ls_i1,k1_i1] + theta_aux_i1.tolist()
    theta_i2 = [k_i2,tc_i2,a,b,q1_i2,q2_i2,ls_i2,k1_i2] + theta_aux_i2.tolist()

    if sub == 'i1':
        return theta_i1
    elif sub == 'i2':
        return theta_i2


class FitSpz2(Fit):

    def __init__(self, setup, data_i1, data_i2, aux1=None, aux2=None, out_dir=None):

        self._setup = setup
        self._out_dir = out_dir
        self._tr  = setup['transit']

        self._data_i1 = data_i1
        self._data_i2 = data_i2

        self._k_i1 = self._tr['k']
        self._k_i2 = self._tr['k']
        self._tc_i1 = self._data_i1['t'].mean()
        self._tc_i2 = self._data_i2['t'].mean()
        self._t14 = self._tr['t14']
        self._p = self._tr['p']
        self._b = 0

        if aux1 is None:
            n = self._data_i1.shape[0]
            bias = np.repeat(1, n)
            aux1 = bias.reshape(1, n)
        self._aux1 = aux1

        if aux2 is None:
            n = self._data_i2.shape[0]
            bias = np.repeat(1, n)
            aux2 = bias.reshape(1, n)
        self._aux2 = aux2

        self._fit_i1 = FitSpz(*data_i1['t f'.split()].values.T, p=self._p, aux=self._aux1)
        self._fit_i2 = FitSpz(*data_i2['t f'.split()].values.T, p=self._p, aux=self._aux2)

        self._logprob = logprob
        self._ld_prior = None

        fp = os.path.join(self._out_dir, 'input.yaml')
        yaml.dump(setup, open(fp, 'w'), default_flow_style=False)


    @property
    def _ini(self):
        k_i1 = self._k_i1
        k_i2 = self._k_i2
        tc_i1 = self._tc_i1
        tc_i2 = self._tc_i2
        p = self._p
        t14 = self._t14
        b = self._b
        q1_i1 = 0.1
        q2_i1 = 0.2
        q1_i2 = 0.1
        q2_i2 = 0.2
        k1_i1 = 0
        k1_i2 = 0
        ls_i1 = np.log(self._data_i1['f'].std())
        ls_i2 = np.log(self._data_i2['f'].std())
        a = util.transit.scaled_a(p, t14, k_i1, np.pi/2)
        pv = [a,b,k_i1,k_i2,tc_i1,tc_i2,q1_i1,q2_i1,q1_i2,q2_i2,ls_i1,ls_i2,k1_i1,k1_i2]
        if self._aux1 is not None:
            pv += [0] * self._aux1.shape[0]
        if self._aux2 is not None:
            pv += [0] * self._aux2.shape[0]
        return np.array(pv)


    @property
    def _args(self):
        lp_i1 = self._fit_i1._logprob
        lp_i2 = self._fit_i2._logprob
        args_i1 = self._fit_i1._args
        args_i2 = self._fit_i2._args
        aux1 = self._aux1
        aux2 = self._aux2
        return lp_i1, lp_i2, args_i1, args_i2, aux1, aux2


    def set_ld_prior(self, ldp_i1=None, ldp_i2=None):
        if ldp_i1 is not None:
            self._fit_i1.set_ld_prior(ldp_i1)
        if ldp_i2 is not None:
            self._fit_i2.set_ld_prior(ldp_i2)


    def post_map(self):

        pv = get_theta(self._pv_map, 'i1', self._aux1, self._aux2)
        fp = os.path.join(self._out_dir, 'map-i1.png')
        self._fit_i1.plot_best(fp=fp, pv=pv)

        pv = get_theta(self._pv_map, 'i2', self._aux1, self._aux2)
        fp = os.path.join(self._out_dir, 'map-i2.png')
        self._fit_i2.plot_best(fp=fp, pv=pv)


    def post_mcmc(self):

        self._make_df_i1()
        fp = os.path.join(self._out_dir, 'spz-i1.csv')
        self._data_i1.to_csv(fp, index=False)

        self._make_df_i2()
        fp = os.path.join(self._out_dir, 'spz-i2.csv')
        self._data_i2.to_csv(fp, index=False)

        fc = self._fc
        names = self._pv_names

        t, f = self._data_i1['t f'.split()].values.T
        ti = np.linspace(t.min(), t.max(), 1000)
        ps = [self._fit_i1.model(pv=get_theta(s, 'i1', self._aux1, self._aux2)) for s in fc[np.random.randint(len(fc), size=100)]]
        fp = os.path.join(self._out_dir, 'mcmc-samples-i1.png')
        plot.samples(t, f, ps, fp=fp)

        t, f = self._data_i2['t f'.split()].values.T
        ps = [self._fit_i2.model(pv=get_theta(s, 'i2', self._aux1, self._aux2)) for s in fc[np.random.randint(len(fc), size=100)]]
        fp = os.path.join(self._out_dir, 'mcmc-samples-i2.png')
        plot.samples(t, f, ps, fp=fp)

        self._plot_corrected()

        self._plot_i1_vs_i2()


    def _make_df_i1(self):

        pv_best = get_theta(self._pv_best, 'i1', self._aux1, self._aux2)
        args_mod = self._fit_i1._args[:-1]
        mod_full = spz_model(pv_best, *args_mod)
        mod_transit = spz_model(pv_best, *args_mod, ret_ma=True)
        mod_sys = spz_model(pv_best, *args_mod, ret_sys=True)
        resid = self._data_i1['f'] - mod_full
        fcor = self._data_i1['f'] - mod_sys
        self._data_i1['f_cor'] = fcor
        self._data_i1['resid'] = resid
        self._data_i1['mod_full'] = mod_full
        self._data_i1['mod_transit'] = mod_transit
        self._data_i1['mod_sys'] = mod_sys


    def _make_df_i2(self):

        pv_best = get_theta(self._pv_best, 'i2', self._aux1, self._aux2)
        args_mod = self._fit_i2._args[:-1]
        mod_full = spz_model(pv_best, *args_mod)
        mod_transit = spz_model(pv_best, *args_mod, ret_ma=True)
        mod_sys = spz_model(pv_best, *args_mod, ret_sys=True)
        resid = self._data_i2['f'] - mod_full
        fcor = self._data_i2['f'] - mod_sys
        self._data_i2['f_cor'] = fcor
        self._data_i2['resid'] = resid
        self._data_i2['mod_full'] = mod_full
        self._data_i2['mod_transit'] = mod_transit
        self._data_i2['mod_sys'] = mod_sys


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


        pv_best_i1 = get_theta(self._pv_best, 'i1', self._aux1, self._aux2)
        resid_i1 = self._fit_i1.resid(pv=pv_best_i1)
        rms_i1 = util.stats.rms(resid_i1)

        tstep_i1 = np.median(np.diff(self._data_i1['t'])) * 86400
        beta_i1 = util.stats.beta(resid_i1, tstep_i1)

        nd, npar = self._data_i1['t'].shape[0], len(pv_best_i1)
        idx = self._pv_names.index('ls_i1')
        sigma_i1 = np.exp(np.median(self._fc[:,idx]))
        rchisq_i1 = util.stats.chisq(resid_i1, sigma_i1, nd, npar, reduced=True)
        bic_i1 = util.stats.bic(self._lp_mcmc, nd, npar)

        summary['i1_rms'] = float(rms_i1)
        summary['i1_beta'] = float(beta_i1)
        summary['i1_rchisq'] = float(rchisq_i1)
        summary['i1_bic'] = float(bic_i1)

        mag, umag = list(map(float, self._oot_phot(self._setup['data'][0], 'i1')))
        summary['i1_mag'] = [mag, umag]


        pv_best_i2 = get_theta(self._pv_best, 'i2', self._aux1, self._aux2)
        resid_i2 = self._fit_i2.resid(pv=pv_best_i2)
        rms_i2 = util.stats.rms(resid_i2)

        tstep_i2 = np.median(np.diff(self._data_i2['t'])) * 86400
        beta_i2 = util.stats.beta(resid_i2, tstep_i2)

        nd, npar = self._data_i2['t'].shape[0], len(pv_best_i2)
        idx = self._pv_names.index('ls_i2')
        sigma_i2 = np.exp(np.median(self._fc[:,idx]))
        rchisq_i2 = util.stats.chisq(resid_i2, sigma_i2, nd, npar, reduced=True)
        bic_i2 = util.stats.bic(self._lp_mcmc, nd, npar)

        summary['i2_rms'] = float(rms_i2)
        summary['i2_beta'] = float(beta_i2)
        summary['i2_rchisq'] = float(rchisq_i2)
        summary['i2_bic'] = float(bic_i2)

        mag, umag = list(map(float, self._oot_phot(self._setup['data'][1], 'i2')))
        summary['i2_mag'] = [mag, umag]

        fp = os.path.join(self._out_dir, 'mcmc-summary.yaml')
        yaml.dump(summary, open(fp, 'w'), default_flow_style=False)


    def _plot_corrected(self, pv=None):

        if pv is None:
            pv = self._pv_best

        t, f = self._data_i1['t f'.split()].values.T
        mod_transit = self._data_i1['mod_transit'].values
        mod_full = self._data_i1['mod_full'].values
        f_cor = self._data_i1['f_cor'].values
        resid = self._data_i1['resid'].values
        fp = os.path.join(self._out_dir, 'fit-best-i1.png')
        corrected_ts(t, f, f_cor, mod_full, mod_transit, resid, fp)

        t, f = self._data_i2['t f'.split()].values.T
        mod_transit = self._data_i2['mod_transit'].values
        mod_full = self._data_i2['mod_full'].values
        f_cor = self._data_i2['f_cor'].values
        resid = self._data_i2['resid'].values
        fp = os.path.join(self._out_dir, 'fit-best-i2.png')
        corrected_ts(t, f, f_cor, mod_full, mod_transit, resid, fp)


    def _plot_i1_vs_i2(self, percs=(50, 16, 84), plot_binned=False):

        fc = self._fc

        args_i1 = self._fit_i1._args
        args_i2 = self._fit_i2._args

        idx = self._pv_names.index('tc_i1')
        tc_i1 = np.median(fc[:,idx])
        t_i1 = self._data_i1['t']
        self._data_i1['phase'] = t_i1 - tc_i1

        idx = self._pv_names.index('tc_i2')
        tc_i2 = np.median(fc[:,idx])
        t_i2 = self._data_i2['t']
        self._data_i2['phase'] = t_i2 - tc_i2

        idx = self._pv_names.index('k_i1')
        k_i1 = fc[:,idx]
        idx = self._pv_names.index('k_i2')
        k_i2 = fc[:,idx]

        npercs = len(percs)

        flux_pr_i1, flux_pr_i2 = [], []
        for theta in fc[np.random.permutation(fc.shape[0])[:1000]]:

            theta_i1 = get_theta(theta, 'i1', self._aux1, self._aux2)
            theta_i2 = get_theta(theta, 'i2', self._aux1, self._aux2)

            flux_pr_i1.append(spz_model(theta_i1, *args_i1[:-1], ret_ma=True))
            flux_pr_i2.append(spz_model(theta_i2, *args_i2[:-1], ret_ma=True))

        flux_pr_i1, flux_pr_i2 = list(map(np.array, [flux_pr_i1, flux_pr_i2]))
        flux_pc_i1 = np.percentile(flux_pr_i1, percs, axis=0)
        flux_pc_i2 = np.percentile(flux_pr_i2, percs, axis=0)

        fp = os.path.join(self._out_dir, 'mcmc-i1-vs-i2.png')

        i1_vs_i2(self._data_i1, self._data_i2, flux_pc_i1, flux_pc_i2,
            npercs, k_i1, k_i2, fp, title=self._plot_title,
            plot_binned=plot_binned)

    @property
    def _plot_title(self):

        if 'name' in list(self._setup['target'].keys()):
            title = self._setup['target']['name']
        else:
            prefix = self._setup['target']['prefix']
            starid = self._setup['target']['starid']
            planet = self._setup['target']['planet']
            title = '{}-{}{}'.format(prefix, starid, planet)
        return title


    def _oot_phot(self, ds, sub):

        channel = ds['channel']
        aor = ds['aor']
        data_dir = ds['datadir']
        fp = os.path.join(data_dir, '{}_phot.pkl'.format(aor))
        cornichon = pickle.load(open(fp, 'rb'))

        pv = get_theta(self._pv_best, sub, self._aux1, self._aux2)
        if sub == 'i1':
            args_mod = self._fit_i1._args
        elif sub == 'i2':
            args_mod = self._fit_i2._args
        t = args_mod[0]
        ti = np.linspace(t.min()-0.5, t.max()+0.5, 10000)
        mod = spz_model(pv, ti, *args_mod[1:-1], ret_ma=True)

        t1, t4 = ti[mod<1][0], ti[mod<1][-1]
        r = '3_3_7'
        mag, umag = sxp.phot.oot_phot(cornichon, t1, t4, r, channel, verbose=False)

        return mag, umag
