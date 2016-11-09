
#!/usr/bin/env python
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
from photutils.morphology import centroid_com, centroid_2dg
import seaborn as sb
sb.set_color_codes('muted')
from emcee import MHSampler, EnsembleSampler, PTSampler
from emcee.utils import sample_ball
import corner
from tqdm import tqdm
import sxp
from pytransit import MandelAgol

from ..k2 import loglike1 as k2_loglike
from like import loglike as spz_loglike
from like import model as spz_model
import util
import plot


METHODS = 'cen pld base'.split()

# import logging
# logger = logging.getLogger('scope.name')
# file_log_handler = logging.FileHandler('logfile.log')
#
# logger.addHandler(file_log_handler)
# stderr_log_handler = logging.StreamHandler()
# logger.addHandler(stderr_log_handler)
#
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_log_handler.setFormatter(formatter)
# stderr_log_handler.setFormatter(formatter)
#
# logger.info('Info message')
# logger.error('Error message')


def logprob(theta, t, f, s, p, aux, k2data, u_kep, u_spz):

    a,i,k_s,k_k,tc_s,tc_k,u1_s,u2_s,u1_k,u2_k,k0_s,k1_s,k0_k,s_k = theta[:14]
    theta_aux = theta[14:]

    if k_s < 0 or k_k < 0 or \
        tc_s < t.min() or tc_s > t.max() or \
        i > np.pi/2:
        return -np.inf
    lp = np.log(stats.norm.pdf(u1_s, u_spz[0], u_spz[1]))
    lp += np.log(stats.norm.pdf(u2_s, u_spz[2], u_spz[3]))
    lp += np.log(stats.norm.pdf(u1_k, u_kep[0], u_kep[1]))
    lp += np.log(stats.norm.pdf(u2_k, u_kep[2], u_kep[3]))

    theta_sp = [k_s,tc_s,a,i,u1_s,u2_s,k0_s,k1_s] + theta_aux.tolist()
    theta_k2 =  k_k,tc_k,a,i,u1_k,u2_k,k0_k,s_k

    ll = spz_loglike(theta_sp, t, f, s, p, aux)
    ll += k2_loglike(theta_k2, k2data[0], k2data[1], p)

    if np.isnan(ll).any():
        return -np.inf
    return ll + lp


def get_theta(theta, sub):

    a,i,k_s,k_k,tc_s,tc_k,u1_s,u2_s,u1_k,u2_k,k0_s,k1_s,k0_k,s2_k = theta[:14]
    theta_aux = theta[14:]
    theta_sp = [k_s,tc_s,a,i,u1_s,u2_s,k0_s,k1_s] + theta_aux.tolist()
    theta_k2 =  k_k,tc_k,a,i,u1_k,u2_k,k0_k,s2_k

    if sub == 'sp':
        return theta_sp
    elif sub == 'k2':
        return theta_k2


class Fit(object):

    """
    Data container and model fitting object for simultaneous analysis of
    K2 and Spitzer data.

    :param dict setup: Parsed from setup YAML file.
    :param str out_dir: Where to save all the output.
    :param str method: One of [base, cen, pld].
    :param int bin_size: Desired bin size for Spitzer data, in seconds.
    :param str k2_kolded_fp: Path to phased-folded K2 light curve.

    setup ::
    """

    def __init__(self, setup, out_dir, method='base', bin_size=60, k2_kolded_fp=None):

        self._setup = setup
        self._out_dir = out_dir
        self._tr  = setup['transit']
        self._pv_best_map = None
        self._logprob_best_map = None
        self._pv_best_mcmc = None
        self._logprob_best_mcmc = None
        self._logprob = logprob
        self._method = method

        fp = os.path.join(out_dir, 'input.yaml')
        yaml.dump(setup, open(fp, 'w'))

        if self._tr['i'] > np.pi/2.:
            self._tr['i'] = np.pi - self._tr['i']
        for k,v in self._tr.items():
            print "{} = {}".format(k,v)

        # setup limb darkening priors
        teff, uteff = setup['stellar']['teff']
        logg, ulogg = setup['stellar']['logg']
        self._u_kep, self._u_spz = util.get_ld(teff, uteff, logg, ulogg)

        # load spitzer data
        self._radius = setup['config']['radius']
        self._aor = setup['config']['aor']
        self._data_dir = setup['config']['data_dir']
        fp = os.path.join(self._data_dir, self._aor+'_phot.pkl')
        df_sp = sxp.util.df_from_pickle(fp, self._radius, pix=True)

        # plot
        fp = os.path.join(out_dir, 'spz_raw.png')
        plot.errorbar(df_sp.t, df_sp.f, df_sp.s, fp, alpha=0.5)

        # extract time series and bin
        keys = ['p{}'.format(i) for i in range(9)]
        pix = df_sp[keys].values
        t, f, s = df_sp.t, df_sp.f, df_sp.s
        timestep = np.median(np.diff(t)) * 24 * 3600
        bs_sec = bin_size
        bs = int(round(bs_sec/timestep))
        binned = functools.partial(util.binned, binsize=bs)
        tb, fb, ub, pixb = map(binned, [t, f, s, pix])
        ub /= np.sqrt(bs)
        t, f, s, pix = tb, fb, ub, pixb
        d_sp = {k:v for k,v in zip('t f s'.split(), [t, f, s])}
        self._df_sp = pd.DataFrame(d_sp)
        self._pix = pix

        # plot
        fp = os.path.join(out_dir, 'spz_binned.png')
        plot.errorbar(tb, fb, ub, fp)

        cube = pix.reshape(-1,3,3)
        cubestacked = np.median(cube, axis=0)
        fp = os.path.join(out_dir, 'spz_pix.png')
        plot.pixels(cubestacked, fp)

        # compute and plot centroids
        cx, cy = centroid_2dg(cubestacked)
        print "cube centroids: {}, {}".format(cx, cy)
        cx, cy = map(int, map(round, [cx, cy]))

        self._xy = np.array([centroid_com(i) for i in cube])
        x, y = self._xy.T
        fp = os.path.join(out_dir, 'spz_cen.png')
        plot.centroids(t, x, y, fp)

        # load k2 data
        names = open(k2_kolded_fp).readline().split(',')
        if len(names) == 3:
            self._df_k2 = pd.read_csv(k2_kolded_fp, names='t f s'.split())
        else:
            self._df_k2 = pd.read_csv(k2_kolded_fp, names='t f'.split())
        fp = os.path.join(out_dir, 'k2_folded.png')
        plot.simple_ts(self._df_k2.t, self._df_k2.f, fp, color='b')

        # set up auxiliary regressors
        if method == 'cen':
            self._aux = self._xy.T
        elif method == 'pld':
            self._aux = pix.T
        elif method == 'base':
            self._aux = None
        else:
            raise ValueError('method must be one of: {}'.format(METHODS))


    def _k2_sig(self):

        """
        Out of transit scatter of the K2 data.
        """

        t, f = self._df_k2[['t','f']].values.T
        t14 = self._tr['t14']
        idx = (t < -t14/2.) | (t > t14/2.)
        sig = f[idx].std()

        return sig


    def _spz_tc(self):

        """
        Uneducated guess for Tc of Spitzer data.
        """

        t = self._df_sp['t']

        return t.mean()


    def _spz_ts(self):

        """
        Spitzer photometric time series (time, flux, unc).
        """

        cols = 't f s'.split()
        t, f, s = self._df_sp[cols].values.T

        return t, f, s


    def _args(self):

        """
        Additional arguments passed to logprob function.
        """

        t, f, s = self._spz_ts()
        p = self._tr['p']
        aux = self._aux
        k2data = self._df_k2[['t','f']].values.T
        u_kep, u_spz = self._u_kep, self._u_spz

        return t, f, s, p, aux, k2data, u_kep, u_spz


    def _initial(self):

        """
        Initial guess parameter vector.
        """

        n_aux = self._aux.shape[0] if self._aux is not None else 0
        s_k = self._k2_sig()
        a, i, k = self._tr['a'], self._tr['i'], self._tr['k']
        tc_s = self._spz_tc()
        tc_k = 0
        u1_s, u2_s = self._u_spz[0], self._u_spz[2]
        u1_k, u2_k = self._u_kep[0], self._u_kep[2]
        k0_s, k1_s, k0_k = 0, 0, 0

        initial = [a, i, k, k, tc_s, tc_k, u1_s, u2_s,
            u1_k, u2_k, k0_s, k1_s, k0_k, s_k]

        initial += [0] * n_aux

        return np.array(initial)


    def _map(self, method='nelder-mead'):

        """
        Maximum a posteriori model fit.
        """

        nlp = lambda *x: -self._logprob(*x)
        initial = self._initial()
        args = self._args()
        res = op.minimize(nlp, initial, args=args, method=method)

        return res


    def max_apo(self, methods=('nelder-mead', 'powell')):

        """
        Attempt maximum a posteriori model fit using both Powell and Nelder-Mead.
        """

        results = []
        for method in methods:
            res = self._map(method=method)
            if res.success:
                print "{} negative log probability: {}".format(method, res.fun)
                results.append(res)
        if len(results) > 0:
            idx = np.argmin([r.fun for r in results])
            map_best = np.array(results)[idx]
            self._pv_best_map = map_best.x
            self._logprob_best_map = -map_best.fun
            self._max_apo_alg = np.array(methods)[idx]
        else:
            print "All methods failed to converge."


    def plot_max_apo(self, fp=None):

        """
        Plot the result of max_apo().
        """

        t, f, s = self._spz_ts()
        initial = self._initial()
        args = self._args()
        alg = self._max_apo_alg
        init_sp = get_theta(initial, 'sp')
        best_sp = get_theta(self._pv_best_map, 'sp')

        init_model = spz_model(init_sp, *args[:-3])
        max_apo_model = spz_model(best_sp, *args[:-3])
        fcor = spz_model(best_sp, *args[:-3], ret_sys=True)
        transit_model = spz_model(best_sp, *args[:-3], ret_ma=True)

        with sb.axes_style('white'):

            fig, axs = pl.subplots(1, 2, figsize=(10,3), sharex=True, sharey=True)
            axs[0].plot(t, f, 'k.')
            axs[0].plot(t, init_model, 'b-', lw=5, label='initial')
            axs[0].plot(t, max_apo_model, 'r-', lw=5, label='optimized')
            axs[0].legend(loc=4)
            axs[1].plot(t, f - fcor, 'k.')
            axs[1].plot(t, transit_model, 'r-', lw=5)

            pl.setp(axs, xlim=[t.min(), t.max()], xticks=[], yticks=[])
            xl, yl = axs[0].get_xlim(), axs[1].get_ylim()
            axs[0].text(xl[0]+0.1*np.diff(xl), yl[0]+0.1*np.diff(yl), alg)
            pl.setp(axs[0], title='raw')
            pl.setp(axs[1], title='corrected')

            fig.tight_layout()
            if fp is None:
                fp = os.path.join(self._out_dir, 'fit-map.png')
            fig.savefig(fp)
            pl.close()


    def run_mcmc(self, nthreads=4, nsteps1=1000, nsteps2=1000, max_steps=1e4,
        gr_threshold=1.1, save=False, restart=False):

        """
        Run MCMC.
        """

        out_dir = self._out_dir

        ndim = len(self._pv_best_map)
        nwalkers = 8 * ndim if ndim > 12 else 16 * ndim
        print "{} walkers exploring {} dimensions".format(nwalkers, ndim)

        args = self._args()

        fp = os.path.join(out_dir, 'flatchain.npz')
        if os.path.isfile(fp) and not restart:

            print "using chain from previous run"
            npz = np.load(fp)
            self._fc = npz['flat_chain']
            self._pv_best_mcmc = npz['pv_best']
            self._logprob_best_mcmc = npz['logprob_best']

        else:

            sampler = EnsembleSampler(nwalkers, ndim, logprob,
                args=args, threads=nthreads)
            pos0 = sample_ball(self._pv_best_map, [1e-5]*ndim, nwalkers)
            pos0[13] = np.abs(pos0[13])

            print "\nstage 1"
            for pos,_,_ in tqdm(sampler.sample(pos0, iterations=nsteps1)):
                pass

            labels = 'a,i,k_s,k_k,tc_s,tc_k,u1_s,u2_s,u1_k,u2_k,k0_s,k1_s,k0_k,s_k'.split(',')
            if self._aux is not None:
                labels += ['c{}'.format(i) for i in range(len(self._aux))]
            fp = os.path.join(out_dir, 'chain-initial.png')
            plot.chain(sampler.chain, labels, fp)

            idx = np.argmax(sampler.lnprobability)
            new_best = sampler.flatchain[idx]
            new_prob = sampler.lnprobability.flat[idx]
            best = new_best if new_prob > self._logprob_best_map else self._pv_best_map
            pos = sample_ball(best, [1e-5]*ndim, nwalkers)
            sampler.reset()
            print "\nstage 2"

            nsteps = 0
            gr_vals = []
            while nsteps < max_steps:
                for pos,_,_ in tqdm(sampler.sample(pos, iterations=nsteps2)):
                    pass
                nsteps += nsteps2
                gr = util.gelman_rubin(sampler.chain)
                gr_vals.append(gr)
                msg = "After {} steps\n\tMean G-R: {}\n\tMax G-R: {}"
                print msg.format(nsteps, gr.mean(), gr.max())
                if (gr < gr_threshold).all():
                    break

            fp = os.path.join(out_dir, 'gr.png')
            plot.gr_iter(gr_vals, fp)

            fp = os.path.join(out_dir, 'chain.png')
            plot.chain(sampler.chain, labels, fp)

            burn = nsteps - nsteps2 if nsteps > nsteps2 else 0
            thin = 1
            self._fc = sampler.chain[:,burn::thin,:].reshape(-1, ndim)
            fp = os.path.join(out_dir, 'corner.png')
            plot.corner(self._fc, labels, fp)

            self._logprob_best_mcmc = sampler.lnprobability.flatten().max()
            idx = np.argmax(sampler.lnprobability)
            assert sampler.lnprobability.flat[idx] == self._logprob_best_mcmc
            self._pv_best_mcmc = sampler.flatchain[idx]

            if save:
                fp = os.path.join(out_dir, 'flatchain')
                np.savez_compressed(
                    fp,
                    flat_chain=self._fc,
                    logprob_best=self._logprob_best_mcmc,
                    pv_best=self._pv_best_mcmc,
                    gelman_rubin=np.array(gr_vals)
                    )

            fp = os.path.join(out_dir, 'opt.txt')
            with open(fp, 'w') as o:
                o.write("MAP log prob: {}".format(self._logprob_best_map))
                o.write("\n\tparams: ")
                o.write(' '.join([str(i) for i in self._pv_best_map]))
                o.write("\nMCMC log prob: {}".format(self._logprob_best_mcmc))
                o.write("\n\tparams: ")
                o.write(' '.join([str(i) for i in self._pv_best_mcmc]))

            best_sp = get_theta(self._pv_best_mcmc, 'sp')
            mod_full = spz_model(best_sp, *args[:-3])
            t, f, s = self._spz_ts()
            sys = spz_model(best_sp, *args[:-3], ret_sys=True)
            f_cor = f - sys
            mod_transit = spz_model(best_sp, *args[:-3], ret_ma=True)
            resid = f - spz_model(best_sp, *args[:-3])
            fp = os.path.join(out_dir, 'fit-mcmc-best.png')
            plot.corrected_ts(t, f, f_cor, mod_full, mod_transit, resid, fp)

            timestep = np.median(np.diff(t)) * 86400
            rms = util.rms(resid)
            beta = util.beta(resid, timestep)
            print "RMS: {}".format(rms)
            print "Beta: {}".format(beta)
            fp = os.path.join(out_dir, 'stats.txt')
            with open(fp, 'w') as o:
                o.write("Method: {}\n".format(self._method))
                o.write("RMS: {}\n".format(rms))
                o.write("Beta: {}\n".format(beta))

        if self._logprob_best_mcmc > self._logprob_best_map:
            best = self._pv_best_mcmc
        else:
            best = self._pv_best_map
        best_sp = get_theta(best, 'sp')
        sys = spz_model(best_sp, *args[:-3], ret_sys=True)
        self._df_sp['f_cor'] = self._df_sp['f'] - sys
        resid = self._df_sp['f'] - spz_model(best_sp, *args[:-3])
        self._df_sp['resid'] = resid
        fp = os.path.join(out_dir, 'spz.csv')
        self._df_sp.to_csv(fp, index=False)


    def plot_final(self):

        """
        Make publication-ready plot.
        """

        t, f, s = self._spz_ts()
        args = self._args()
        fc = self._fc
        tc = np.median(fc[:,4])
        df_sp = self._df_sp
        df_sp['phase'] = t - tc
        df_k2 = self._df_k2
        p = self._tr['p']

        percs = [50, 16, 84]
        npercs = len(percs)

        df_k2.ti = np.linspace(df_k2.t.min(), df_k2.t.max(), 1000)

        flux_pr_k2, flux_pr_sp = [], []
        for theta in fc[np.random.permutation(fc.shape[0])[:1000]]:

            theta_sp = get_theta(theta, 'sp')
            theta_k2 = get_theta(theta, 'k2')

            flux_pr_sp.append(spz_model(theta_sp, *args[:-3], ret_ma=True))
            flux_pr_k2.append(k2_loglike(theta_k2, df_k2.ti, df_k2['f'], p, ret_mod=True))

        flux_pr_sp, flux_pr_k2 = map(np.array, [flux_pr_sp, flux_pr_k2])
        flux_pc_sp = np.percentile(flux_pr_sp, percs, axis=0)
        flux_pc_k2 = np.percentile(flux_pr_k2, percs, axis=0)

        fp = os.path.join(self._out_dir, 'fit-final.png')
        title = self._setup['config']['star'] + self._setup['config']['planet']
        plot.k2_spz_together(self._df_sp, self._df_k2, flux_pc_sp, flux_pc_k2,
            percs, self._fc[:,2], self._fc[:,3], fp, title=title)
