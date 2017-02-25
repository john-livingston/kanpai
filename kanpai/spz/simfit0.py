import os
import sys
import yaml
import pickle

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.optimize as op
from scipy import stats
import seaborn as sb
from emcee import MHSampler, EnsembleSampler, PTSampler
from emcee.utils import sample_ball
import corner
from tqdm import tqdm
from pytransit import MandelAgol


import plot
from .like import loglike_u as spz_loglike
from .mod import model_u as spz_model
from .util import setup_aux
from io import load_spz
from ld import get_ld_claret
from .. import util
from ..k2.like import loglike_u as k2_loglike
from ..k2 import fit as k2_fit
from ..k2 import plot as k2_plot
from ..k2 import ld as k2_ld
from ..k2.io import load_k2
from ..engines import MAP, MCMC
from ..plot import multi_gauss_fit

np.warnings.simplefilter('ignore')
sb.set_color_codes('muted')


K2_TIME_OFFSET = 2454833
PI2 = np.pi/2


def logprob(theta, t, f, p, aux, k2data, u_kep, u_spz, ret_pvnames=False):

    if ret_pvnames:
        pvn = 'a,b,k_s,k_k,tc_s,tc_k,u1_s,u2_s,u1_k,u2_k,s_s,s_k,k1_s,k0_k'.split(',')
        if aux is not None:
            pvn += ['c{}'.format(i) for i in range(len(aux))]
        return pvn

    a,b,k_s,k_k,tc_s,tc_k,u1_s,u2_s,u1_k,u2_k,s_s,s_k,k1_s,k0_k = theta[:14]
    theta_aux = theta[14:]

    if k_s < -1 or k_s > 1 or k_k < -1 or k_k > 1 or \
        tc_s < t[0] - 0.05 or tc_s > t[-1] + 0.05 or \
        s_s < 0 or s_s > 1 or s_k < 0 or s_k > 1 or \
        b < 0 or b > 1:
        return -np.inf
    lp = np.log(stats.norm.pdf(u1_s, u_spz[0], u_spz[1]))
    lp += np.log(stats.norm.pdf(u2_s, u_spz[2], u_spz[3]))
    lp += np.log(stats.norm.pdf(u1_k, u_kep[0], u_kep[1]))
    lp += np.log(stats.norm.pdf(u2_k, u_kep[2], u_kep[3]))

    theta_sp = [k_s,tc_s,a,b,u1_s,u2_s,s_s,k1_s] + theta_aux.tolist()
    theta_k2 =  k_k,tc_k,a,b,u1_k,u2_k,s_s,k0_k

    ll = spz_loglike(theta_sp, t, f, p, aux)
    ll += k2_loglike(theta_k2, k2data[0], k2data[1], p)

    if np.isnan(ll).any():
        return -np.inf
    return ll + lp


def get_theta(theta, sub):

    a,b,k_s,k_k,tc_s,tc_k,u1_s,u2_s,u1_k,u2_k,s_s,s_k,k1_s,k0_k = theta[:14]
    theta_aux = theta[14:]
    theta_sp = [k_s,tc_s,a,b,u1_s,u2_s,s_s,k1_s] + theta_aux.tolist()
    theta_k2 =  k_k,tc_k,a,b,u1_k,u2_k,s_k,k0_k

    if sub == 'sp':
        return theta_sp
    elif sub == 'k2':
        return theta_k2


class Fit(object):

    """
    Data container and model fitting object for simultaneous analysis of
    K2 and Spitzer data.

    :param dict setup   : Parsed from setup YAML file.
    :param str out_dir  : Where to save all the output.
    :param str method   : One of [base, cen, pld].
    :param int bin_size : Desired bin size for Spitzer data, in seconds.

    """

    def __init__(self, setup, out_dir, method='base'):

        self._setup = setup
        self._out_dir = out_dir
        self._tr  = setup['transit']
        self._logprob = logprob
        self._method = method
        self._pv_map = None
        self._lp_map = None
        self._max_apo_alg = None
        self._pv_mcmc = None
        self._lp_mcmc = None
        self._output = dict(method=method)

        fp = os.path.join(out_dir, 'input.yaml')
        yaml.dump(setup, open(fp, 'w'), default_flow_style=False)

        print "\nInitial parameter values:"
        for k,v in self._tr.items():
            print "{} = {}".format(k,v)

        # setup limb darkening priors
        self._setup_ld()

        # load k2 data
        self._load_k2()

            # load spitzer data
        self._load_spz()

        # set up auxiliary regressors
        self._setup_aux()


    def _setup_aux(self):

        self._aux = setup_aux(self._method, self._xy, self._pix)


    def _setup_ld(self):

        teff, uteff = self._setup['stellar']['teff']
        logg, ulogg = self._setup['stellar']['logg']
        feh, ufeh = self._setup['stellar']['feh']

        try:
            self._u_kep = self._setup['ld']['kep']
        except KeyError as e:
            print "Input missing Kepler limb-darkening priors"
            print "Using LDTk..."
            self._u_kep = k2_ld.get_ld_ldtk(teff, uteff, logg, ulogg, feh, ufeh)

        try:
            self._u_spz = self._setup['ld']['spz']
        except KeyError as e:
            print "Input missing Spitzer limb-darkening priors"
            print "Using Claret+2012..."
            self._u_spz = get_ld_claret(teff, uteff, logg, ulogg, 'S2')

        self._output['ld_priors'] = dict(kep=self._u_kep, spz=self._u_spz)


    def _load_k2(self):

        k2_folded_fp = self._setup['config']['k2lc']
        binning = self._setup['config']['bin_k2'] if 'bin_k2' in self._setup['config'].keys() else None
        self._df_k2 = load_k2(k2_folded_fp, binning)


    def _load_spz(self):

        df, pix = load_spz(self._setup, self._out_dir)
        self._df_sp = df
        self._pix = pix
        self._xy = df[['x', 'y']].values

        # self._radius = self._setup['config']['radius']
        # self._aor = self._setup['config']['aor']
        # self._data_dir = self._setup['config']['datadir']
        # self._geom = self._setup['config']['geom']
        # if self._geom == '3x3':
        #     self._npix = 9
        # elif self._geom == '5x5':
        #     self._npix = 25




    def _add_k2_sig(self):

        t, f = self._df_k2[['t','f']].values.T
        t14 = self._tr['t14']
        idx = (t < -t14/2.) | (t > t14/2.)
        sig = f[idx].std()
        self._df_k2['s'] = np.repeat(sig, f.size)



    @property
    def _spz_tc(self):

        """
        Optimistic guess for Tc of Spitzer data.
        """

        t = self._df_sp['t']

        return t.mean()


    @property
    def _spz_ts(self):

        """
        Spitzer photometric time series (time, flux, unc).
        """

        cols = 't f s'.split()
        t, f, s = self._df_sp[cols].values.T

        return t, f, s


    @property
    def _spz_args(self):

        """
        Spitzer likelihood args.
        """

        t, f, s = self._spz_ts
        p = self._tr['p']
        aux = self._aux

        return t, f, p, aux


    @property
    def _k2_ts(self):

        """
        K2 photometric time series (time, flux, unc).
        """

        cols = 't f s'.split()
        t, f, s = self._df_k2[cols].values.T

        return t, f, s


    @property
    def _k2_args(self):

        """
        K2 likelihood args.
        """

        t, f, s = self._k2_ts
        p = self._tr['p']

        return t, f, p


    @property
    def _args(self):

        """
        Additional arguments passed to logprob function.
        """

        t, f, s = self._spz_ts
        p = self._tr['p']
        aux = self._aux
        k2data = self._df_k2[['t','f','s']].values.T
        u_kep, u_spz = self._u_kep, self._u_spz

        return t, f, p, aux, k2data, u_kep, u_spz


    @property
    def _ini(self):

        """
        Initial guess parameter vector.
        """

        n_aux = self._aux.shape[0] if self._aux is not None else 0
        a, b, k = self._tr['a'], self._tr['b'], self._tr['k']
        tc_s = self._spz_tc
        tc_k = 0
        u1_s, u2_s = self._u_spz[0], self._u_spz[2]
        u1_k, u2_k = self._u_kep[0], self._u_kep[2]
        s_s = self._df_sp['f'].std()
        s_k = self._df_k2['f'].std()
        k1_s, k0_k = 0, 0

        initial = [a, b, k, k, tc_s, tc_k, u1_s, u2_s,
            u1_k, u2_k, s_s, s_k, k1_s, k0_k]

        initial += [0] * n_aux

        return np.array(initial)


    @property
    def _labels(self):
        # FIXME hacky
        labels = self._logprob(self._ini, *self._args, ret_pvnames=True)
        return labels


    def _pv_names(self, idx):

        """
        Parameter names for a given index.
        """

        pvna = np.array(self._labels)
        return pvna[idx]


    def _pn_idx(self, pname):

        """
        Index for a given parameter name.
        """

        return self._labels.index(pname)


    def run_map(self, methods=('nelder-mead', 'powell'), make_plots=True):

        """
        Run a maximum a posteriori (MAP) fit using one or more methods.
        Defaults to Nelder-Mead and Powell.
        """

        self._map = MAP(self._logprob, self._ini, self._args, methods=methods)
        self._map.run()
        self._pv_map, self._lp_map, self._max_apo_alg = self._map.results

        if self._pv_map is None:
            self._pv_map = self._ini
            self._lp_map = self._logprob(self._pv_map, *self._args)
            self._max_apo_alg = 'none'

        delta = np.abs(self._ini / self._pv_map)
        threshold = 2
        idx = ( (delta > threshold) | ((delta < 1./threshold) & (delta != 0)) )
        if idx.any():
            print "WARNING -- some MAP parameters changed by more than 2x:"
            print self._pv_names(idx)
            print "Overriding MAP parameters with initial guesses"
            self._pv_map = self._ini
            self._lp_map = self._logprob(self._pv_map, *self._args)

        if 'opt' not in self._output.keys():
            self._output['opt'] = {}
        self._output['opt']['map'] = dict(
            logprob=float(self._lp_map),
            pv=dict(zip(self._labels, self._pv_map.tolist()))
            )

        if make_plots:
            self._plot_max_apo()


    def _plot_max_apo(self):

        """
        Plot the result of max_apo().
        """

        t, f, s = self._spz_ts
        initial = self._ini
        args = self._args
        alg = self._max_apo_alg
        init_sp = get_theta(initial, 'sp')
        best_sp = get_theta(self._pv_map, 'sp')

        init_model = spz_model(init_sp, *self._spz_args)
        max_apo_model = spz_model(best_sp, *self._spz_args)
        fcor = spz_model(best_sp, *self._spz_args, ret_sys=True)
        transit_model = spz_model(best_sp, *self._spz_args, ret_ma=True)

        rc = {'xtick.direction': 'in',
              'ytick.direction': 'in',
              'xtick.major.size': 5,
              'ytick.major.size': 5}

        with sb.axes_style('white', rc):

            fig, axs = pl.subplots(1, 2, figsize=(10,3), sharex=True, sharey=True)
            axs[0].plot(t, f, 'k.')
            axs[0].plot(t, init_model, 'b-', lw=2, label='initial')
            axs[0].plot(t, max_apo_model, 'r-', lw=1.5, label='optimized')
            axs[0].legend(loc=4)
            axs[1].plot(t, f - fcor, 'k.')
            axs[1].plot(t, transit_model, 'r-', lw=3)

            axs.flat[0].xaxis.get_major_formatter().set_useOffset(False)
            axs.flat[1].xaxis.get_major_formatter().set_useOffset(False)
            axs.flat[0].yaxis.get_major_formatter().set_useOffset(False)
            axs.flat[1].yaxis.get_major_formatter().set_useOffset(False)
            pl.setp(axs.flat[0].xaxis.get_majorticklabels(), rotation=20)
            pl.setp(axs.flat[1].xaxis.get_majorticklabels(), rotation=20)

            pl.setp(axs, xlim=[t.min(), t.max()])
            xl, yl = axs[0].get_xlim(), axs[1].get_ylim()
            axs[0].text(xl[0]+0.1*np.diff(xl), yl[0]+0.1*np.diff(yl), alg)
            pl.setp(axs.flat[0], title='Raw data', ylabel='Normalized flux')
            pl.setp(axs.flat[1], title='Corrected', ylabel='Normalized flux')
            pl.setp(axs, xlim=[t.min(), t.max()], xlabel='Time [BJD]')

            fig.tight_layout()
            fp = os.path.join(self._out_dir, 'fit-map.png')
            fig.savefig(fp)
            pl.close()


    def run_mcmc(self, save=False, restart=False, resume=False, make_plots=True):

        """
        Run MCMC.
        """

        nthreads = self._setup['config']['nthreads']
        nsteps1 = self._setup['config']['nsteps1']
        nsteps2 = self._setup['config']['nsteps2']
        max_steps = self._setup['config']['maxsteps']
        gr_threshold = self._setup['config']['grthreshold']
        out_dir = self._out_dir
        args = self._args
        logprob = self._logprob
        names = self._labels

        if self._pv_map is None:
            pv_ini = self._ini
            logprob_ini = logprob(pv_ini, *args)
        else:
            pv_ini = self._pv_map
            logprob_ini = self._lp_map

        fp = os.path.join(out_dir, 'mcmc.npz')
        if os.path.isfile(fp):

            if resume:

                print "Resuming from previous best position"
                npz = np.load(fp)
                pv_ini = npz['pv_best']
                logprob_ini = npz['logprob_best']

            elif not restart:

                print "Loading chain from previous run"
                npz = np.load(fp)
                self._fc = npz['flat_chain']
                self._pv_mcmc = npz['pv_best']
                self._lp_mcmc = npz['logprob_best']

                self._output['opt']['mcmc'] = dict(
                    logprob=float(self._lp_mcmc),
                    pv=dict(zip(self._labels, self._pv_mcmc.tolist()))
                    )

                self._update_df_sp()
                fp = os.path.join(self._out_dir, 'spz.csv')
                self._df_sp.to_csv(fp, index=False)
                self._post_mcmc()

                return

        self._mcmc = MCMC(logprob, pv_ini, args, names, out_dir)
        self._mcmc.run(nthreads, nsteps1, nsteps2, max_steps, gr_threshold, save=True, make_plots=True)
        pv, lp, fc, gr, acor = self._mcmc.results
        self._pv_mcmc, self._lp_mcmc, self._fc = pv, lp, fc
        self._update_df_sp()

        if 'opt' not in self._output.keys():
            self._output['opt'] = {}

        self._output['opt']['mcmc'] = dict(
            logprob=float(self._lp_mcmc),
            pv=dict(zip(self._labels, self._pv_mcmc.tolist()))
            )

        self._output['stats'] = dict(
            gr=dict(zip(self._labels, gr.tolist()))
            )
        if acor is not None:
            self._output['stats']['acor']=dict(zip(self._labels, acor.tolist()))

        self._post_mcmc()


    @property
    def _pv_best(self):
        if self._lp_mcmc > self._lp_map:
            best = self._pv_mcmc
        else:
            best = self._pv_map
        return best


    def _plot_best(self):
        t, f, s = self._spz_ts
        mod_transit = self._df_sp['mod_transit'].values
        mod_full = self._df_sp['mod_full'].values
        f_cor = self._df_sp['f_cor'].values
        resid = self._df_sp['resid'].values
        fp = os.path.join(self._out_dir, 'fit-best.png')
        plot.corrected_ts(t, f, f_cor, mod_full, mod_transit, resid, fp)


    def _post_mcmc(self):

        self._plot_best()

        # Spitzer stats
        t = self._df_sp['t']
        s = self._df_sp['s']
        resid = self._df_sp['resid']
        best_sp = get_theta(self._pv_best, 'sp')
        timestep = np.median(np.diff(t)) * 86400
        rms = util.stats.rms(resid)
        beta = util.stats.beta(resid, timestep)
        nd, npar = len(t), len(best_sp)
        rchisq = util.stats.chisq(resid, s, nd, npar, reduced=True)
        obj = lambda x: (1 - util.stats.chisq(resid, s*x, nd, npar, reduced=True))**2
        res = op.minimize(obj, 1, method='nelder-mead')
        if res.success:
            rescale_fac = float(res.x)
        else:
            rescale_fac = None
        bic = util.stats.bic(spz_loglike(best_sp, *self._spz_args), nd, npar)

        print "RMS: {}".format(rms)
        print "Beta: {}".format(beta)
        self._output['spz'] = dict(rms=float(rms),
            beta=float(beta),
            reduced_chisq=float(rchisq),
            rescale_fac=rescale_fac,
            bic=float(bic)
            )

        # K2 stats
        best_k2 = get_theta(self._pv_best, 'k2')
        df_k2 = self._df_k2
        t, f, s = df_k2['t'], df_k2['f'], df_k2['s']
        model = k2_loglike(best_k2, *self._k2_args, ret_mod=True)
        resid = f - model
        rms = util.stats.rms(resid)
        timestep = np.median(np.diff(t)) * 86400
        try:
            beta = float(util.stats.beta(resid, timestep))
        except:
            beta = None
        nd, npar = len(s), len(best_k2)
        rchisq = util.stats.chisq(resid, s, nd, npar, reduced=True)
        obj = lambda x: (1 - util.stats.chisq(resid, s*x, nd, npar, reduced=True))**2
        res = op.minimize(obj, 1, method='nelder-mead')
        if res.success:
            rescale_fac = float(res.x)
        else:
            rescale_fac = None
        bic = util.stats.bic(k2_loglike(best_k2, *self._k2_args), nd, npar)

        self._output['k2'] = dict(rms=float(rms),
            beta=beta,
            reduced_chisq=float(rchisq),
            rescale_fac=rescale_fac,
            bic=float(bic)
            )

        # percentiles
        percs = [15.87, 50.0, 84.13]
        pc = np.percentile(self._fc, percs, axis=0).T.tolist()
        self._output['percentiles'] = dict(zip(self._labels, pc))

        # rhostar
        p = self._tr['p']
        idx = self._pn_idx('a')
        a_samples = self._fc[:,idx]
        rho = util.transit.sample_rhostar(a_samples, p)
        p0 = 1,rho.mean(),rho.std(), 1,np.median(rho),rho.std()
        fp = os.path.join(self._out_dir, 'rhostar.png')
        multi_gauss_fit(rho, p0, fp=fp)

        # small corner
        fp = os.path.join(self._out_dir, 'corner-small.png')
        idx = []
        for n in 'a b k_s k_k s_s s_k tc_s'.split():
            idx += [self._pn_idx(n)]
        fc = self._fc[:,idx].copy()
        tc = int(fc[:,-1].mean())
        fc[:,-1] -= tc
        # fc[:,1] *= 180/np.pi
        labels = r'$a/R_{\star}$ $b$ $R_p/R_{\star,S}$ $R_p/R_{\star,K}$ $\sigma_{S}$ $\sigma_{K}$'
        labels += r' $T_[C,S]-{}$'.format(tc).replace('[','{').replace(']','}')
        plot.corner(fc, labels.split(), fp=fp,
            quantiles=None, plot_datapoints=False, dpi=256, tight=True)


    def _update_df_sp(self):

        args = self._args
        best_sp = get_theta(self._pv_best, 'sp')
        mod_full = spz_model(best_sp, *self._spz_args)
        mod_transit = spz_model(best_sp, *self._spz_args, ret_ma=True)
        mod_sys = spz_model(best_sp, *self._spz_args, ret_sys=True)
        resid = self._df_sp['f'] - mod_full
        fcor = self._df_sp['f'] - mod_sys
        self._df_sp['f_cor'] = fcor
        self._df_sp['resid'] = resid
        self._df_sp['mod_full'] = mod_full
        self._df_sp['mod_transit'] = mod_transit
        self._df_sp['mod_sys'] = mod_sys


    def dump(self):

        self._output['ini'] = {}
        for k,v in self._tr.items():
            if k == 'u':
                continue
            self._output['ini'][k] = float(v)
        fp = os.path.join(self._out_dir, 'output.yaml')
        yaml.dump(self._output, open(fp, 'w'), default_flow_style=False)


    def plot_final(self, percs=(50, 16, 84), plot_binned=False):

        """
        Make publication-ready plot.
        """

        if 'f_cor' not in self._df_sp.columns:
            fp = os.path.join(self._out_dir, 'spz.csv')
            self._df_sp = pd.read_csv(fp)

        t, f, s = self._spz_ts
        args = self._args
        fc = self._fc
        tc = np.median(fc[:,4])
        df_sp = self._df_sp
        df_sp['phase'] = t - tc
        df_k2 = self._df_k2
        p = self._tr['p']

        npercs = len(percs)

        df_k2.ti = np.linspace(df_k2.t.min(), df_k2.t.max(), 1000)
        args_k2 = df_k2.ti, df_k2['f'], p

        flux_pr_k2, flux_pr_sp = [], []
        for theta in fc[np.random.permutation(fc.shape[0])[:1000]]:

            theta_sp = get_theta(theta, 'sp')
            theta_k2 = get_theta(theta, 'k2')

            flux_pr_sp.append(spz_model(theta_sp, *self._spz_args, ret_ma=True))
            flux_pr_k2.append(k2_loglike(theta_k2, *args_k2, ret_mod=True))

        flux_pr_sp, flux_pr_k2 = map(np.array, [flux_pr_sp, flux_pr_k2])
        flux_pc_sp = np.percentile(flux_pr_sp, percs, axis=0)
        flux_pc_k2 = np.percentile(flux_pr_k2, percs, axis=0)

        fp = os.path.join(self._out_dir, 'fit-final.png')

        plot.k2_vs_spz(self._df_sp, self._df_k2, flux_pc_sp, flux_pc_k2,
            npercs, self._fc[:,2], self._fc[:,3], fp, title=self._plot_title,
            plot_binned=plot_binned)


    @property
    def _plot_title(self):

        if 'name' in self._setup['config'].keys():
            title = self._setup['config']['name']
        else:
            prefix = self._setup['config']['prefix']
            starid = self._setup['config']['starid']
            planet = self._setup['config']['planet']
            title = '{}-{}{}'.format(prefix, starid, planet)
        if 'epoch' in self._setup['config'].keys():
            epoch = self._setup['config']['epoch']
            title += ' epoch {}'.format(epoch)
        return title
