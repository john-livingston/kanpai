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
from .plot import k2_vs_spz
from ..spz.fit import FitSpz
from ..k2.fit import FitK2
from ..fit import Fit
from .util import make_quantiles_table


def logprob(theta, lp_k2, lp_spz, args_k2, args_spz, aux=None, ret_pvnames=False):

    if ret_pvnames:
        pvn = 'a,b,k_s,k_k,tc_s,tc_k,q1_s,q2_s,q1_k,q2_k,ls_s,ls_k,k1_s,k0_k'.split(',')
        if aux is not None:
            pvn += ['c{}'.format(i) for i in range(len(aux))]
        return pvn

    theta_k2 = get_theta(theta, 'k2')
    theta_spz = get_theta(theta, 'spz')

    lp = lp_k2(theta_k2, *args_k2) + lp_spz(theta_spz, *args_spz)

    return lp


def get_theta(theta, sub):

    a,b,k_s,k_k,tc_s,tc_k,q1_s,q2_s,q1_k,q2_k,ls_s,ls_k,k1_s,k0_k = theta[:14]
    theta_aux = theta[14:]

    theta_k2 =  [k_k,tc_k,a,b,q1_k,q2_k,ls_k,k0_k]
    theta_sp = [k_s,tc_s,a,b,q1_s,q2_s,ls_s,k1_s] + theta_aux.tolist()

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

        fp = os.path.join(self._out_dir, 'input.yaml')
        yaml.dump(setup, open(fp, 'w'), default_flow_style=False)


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
        ls_k = np.log(self._data_k2['f'].std())
        ls_s = np.log(self._data_spz['f'].std())
        a = util.transit.scaled_a(p, t14, k_k, np.pi/2)
        pv = [a,b,k_s,k_k,tc_s,tc_k,q1_s,q2_s,q1_k,q2_k,ls_s,ls_k,k1_s,k0_k]
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


    def set_ld_prior(self, ldp_k2=None, ldp_spz=None):
        if ldp_k2 is not None:
            self._fit_k2.set_ld_prior(ldp_k2)
        if ldp_spz is not None:
            self._fit_spz.set_ld_prior(ldp_spz)


    def post_map(self):

        pv = get_theta(self._pv_map, 'k2')
        fp = os.path.join(self._out_dir, 'map-k2.png')
        self._fit_k2.plot_best(fp=fp, nmodel=1000, pv=pv)

        pv = get_theta(self._pv_map, 'spz')
        fp = os.path.join(self._out_dir, 'map-spz.png')
        self._fit_spz.plot_best(fp=fp, pv=pv)


    def post_mcmc(self):

        self._make_df_spz()
        fp = os.path.join(self._out_dir, 'spz.csv')
        self._data_spz.to_csv(fp, index=False)

        fc = self._fc
        names = self._pv_names

        t, f = self._data_k2['t f'.split()].values.T
        ti = np.linspace(t.min(), t.max(), 1000)
        ps = [self._fit_k2.model(t=ti,pv=get_theta(s, 'k2')) for s in fc[np.random.randint(len(fc), size=100)]]
        fp = os.path.join(self._out_dir, 'mcmc-samples-k2.png')
        plot.samples(t, f, ps, tmodel=ti, fp=fp)

        t, f = self._data_spz['t f'.split()].values.T
        ps = [self._fit_spz.model(pv=get_theta(s, 'spz')) for s in fc[np.random.randint(len(fc), size=100)]]
        fp = os.path.join(self._out_dir, 'mcmc-samples-spz.png')
        plot.samples(t, f, ps, fp=fp)

        self._plot_corrected()

        self._plot_k2_vs_spz()


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

        pv_best_spz = get_theta(self._pv_best, 'spz')
        resid_spz = self._fit_spz.resid(pv=pv_best_spz)
        rms_spz = util.stats.rms(resid_spz)

        tstep_spz = np.median(np.diff(self._data_spz['t'])) * 86400
        beta_spz = util.stats.beta(resid_spz, tstep_spz)

        nd, npar = self._data_spz['t'].shape[0], len(pv_best_spz)
        idx = self._pv_names.index('ls_s')
        sigma_spz = np.exp(np.median(self._fc[:,idx]))
        rchisq_spz = util.stats.chisq(resid_spz, sigma_spz, nd, npar, reduced=True)
        bic_spz = util.stats.bic(self._lp_mcmc, nd, npar)

        summary['spz_rms'] = float(rms_spz)
        summary['spz_beta'] = float(beta_spz)
        summary['spz_rchisq'] = float(rchisq_spz)
        summary['spz_bic'] = float(bic_spz)

        mag, umag = list(map(float, self._oot_phot()))
        summary['spz_mag'] = [mag, umag]

        fp = os.path.join(self._out_dir, 'mcmc-summary.yaml')
        yaml.dump(summary, open(fp, 'w'), default_flow_style=False)


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


    def _plot_k2_vs_spz(self, percs=(50, 16, 84), plot_binned=False):

        t = self._data_spz['t']
        args = self._args
        fc = self._fc
        idx = self._pv_names.index('tc_s')
        tc = np.median(fc[:,idx])
        self._data_spz['phase'] = t - tc

        idx = self._pv_names.index('k_s')
        k_s = fc[:,idx]
        idx = self._pv_names.index('k_k')
        k_k = fc[:,idx]

        args_k2 = self._fit_k2._args
        args_spz = self._fit_spz._args
        lp_k2 = self._fit_k2._logprob

        npercs = len(percs)

        nmodel_k2 = self._data_k2.shape[0]
        ti_k2 = np.linspace(self._data_k2.t.min(), self._data_k2.t.max(), nmodel_k2)
        self._data_k2['ti'] = ti_k2

        flux_pr_k2, flux_pr_sp = [], []
        for theta in fc[np.random.permutation(fc.shape[0])[:1000]]:

            theta_sp = get_theta(theta, 'spz')
            theta_k2 = get_theta(theta, 'k2')

            flux_pr_sp.append(spz_model(theta_sp, *args_spz[:-1], ret_ma=True))
            flux_pr_k2.append(lp_k2(theta_k2, ti_k2, *args_k2[1:], ret_mod=True))

        flux_pr_sp, flux_pr_k2 = list(map(np.array, [flux_pr_sp, flux_pr_k2]))
        flux_pc_sp = np.percentile(flux_pr_sp, percs, axis=0)
        flux_pc_k2 = np.percentile(flux_pr_k2, percs, axis=0)

        fp = os.path.join(self._out_dir, 'mcmc-k2-vs-spz.png')

        k2_vs_spz(self._data_spz, self._data_k2, flux_pc_sp, flux_pc_k2,
            npercs, k_s, k_k, fp, title=self._plot_title,
            plot_binned=plot_binned)

    @property
    def _plot_title(self):

        if 'name' in list(self._setup['config'].keys()):
            title = self._setup['config']['name']
        else:
            prefix = self._setup['config']['prefix']
            starid = self._setup['config']['starid']
            planet = self._setup['config']['planet']
            title = '{}-{}{}'.format(prefix, starid, planet)
        if 'epoch' in list(self._setup['config'].keys()):
            epoch = self._setup['config']['epoch']
            title += ' epoch {}'.format(epoch)
        return title


    def _oot_phot(self):

        aor = self._setup['config']['aor']
        data_dir = self._setup['config']['datadir']
        fp = os.path.join(data_dir, '{}_phot.pkl'.format(aor))
        cornichon = pickle.load(open(fp, 'rb'))

        pv = get_theta(self._pv_best, 'spz')
        args_mod = self._fit_spz._args[:-1]
        t = self._fit_spz._args[0]
        ti = np.linspace(t.min()-0.5, t.max()+0.5, 10000)
        mod = spz_model(pv, ti, *self._fit_spz._args[1:-1], ret_ma=True)

        t1, t4 = ti[mod<1][0], ti[mod<1][-1]
        r = '3_3_7'
        mag, umag = sxp.phot.oot_phot(cornichon, t1, t4, r=r, verbose=False)

        return mag, umag


    def update_ephemeris(self, t0=None, ut0=None):

        if t0 is None:
            t0 = self._tr['t0']
        if ut0 is None:
            ut0 = self._tr['ut0']

        idx = self._pv_names.index('tc_s')
        tc_s = self._fc[:,idx]
        n = tc_s.shape[0]
        t0_s = np.random.randn(n) * ut0 + t0
        tc_samples = np.c_[t0_s, tc_s].T

        p0 = self._p
        n_orb = int(round( (np.median(tc_s) - t0) / p0 ))
        orb = [0, n_orb]

        ephem_samples = util.transit.sample_ephem(orb, tc_samples)
        fp = os.path.join(self._out_dir, 'ephem-corner.png')
        plot.corner(ephem_samples, labels=r'$T_0$ $P$'.split(),
            # truths=[t0,p0],
            # tight=True,
            fp=fp)

        p_s = ephem_samples[:,1]
        p1, up1 = np.median(p_s), p_s.std()
        msg = "P = {} +/- {}".format(p1, up1)
        print msg
        fp = os.path.join(self._out_dir, 'ephem.txt')
        with open(fp, 'w') as w:
            w.write(msg)

        fp = os.path.join(self._out_dir, 'ephem.npz')
        np.savez_compressed(fp, samples=ephem_samples)

        try:
            up0 = self._tr['up']
            tc_best = self.best['tc_s']
            tc_pred = t0 + n_orb * p0
            utc_pred = np.sqrt(ut0**2 + (n_orb*up0)**2)
            utc = tc_s.std()
            print "{0:.2f}-sigma diff. in Tc from expected".format((tc_pred - tc_best) / np.sqrt(utc_pred**2 + utc**2))
        except:
            pass

        self._p = p1
        self._p_s = p_s


    def make_table(self):
        try:
            fp = os.path.join(self._out_dir, 'mcmc.npz')
            make_quantiles_table(fp, self._p) # FIXME: pass samples (self._fc) instead of fp
        except:
            print "Saved samples not found, not making table."
