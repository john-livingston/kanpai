import os
import sys
import yaml
import pickle
import functools

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.optimize as op
from scipy import stats
from photutils.morphology import centroid_com, centroid_2dg
import seaborn as sb
from emcee import MHSampler, EnsembleSampler, PTSampler
from emcee.utils import sample_ball
import corner
from tqdm import tqdm
import sxp
from pytransit import MandelAgol

from ..k2 import loglike1 as k2_loglike
from ..k2 import fit as k2_fit
from ..k2 import plot as k2_plot
from ..k2 import util as k2_util
from like import loglike as spz_loglike
from like import model as spz_model
import util
import plot

np.warnings.simplefilter('ignore')
sb.set_color_codes('muted')


K2_TIME_OFFSET = 2454833
PI2 = np.pi/2

METHODS = 'cen pld base pca pca2 pca-quad cen-quad pld-quad'.split()

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

    a,i,k_s,k_k,tc_s,tc_k,u1_s,u2_s,u1_k,u2_k,k1_s,k0_k = theta[:12]
    theta_aux = theta[12:]

    if k_s < -1 or k_s > 1 or k_k < -1 or k_k > 1 or \
        tc_s < t[0] - 0.05 or tc_s > t[-1] + 0.05 or \
        i < 0 or i > PI2:
        return -np.inf
    lp = np.log(stats.norm.pdf(u1_s, u_spz[0], u_spz[1]))
    lp += np.log(stats.norm.pdf(u2_s, u_spz[2], u_spz[3]))
    lp += np.log(stats.norm.pdf(u1_k, u_kep[0], u_kep[1]))
    lp += np.log(stats.norm.pdf(u2_k, u_kep[2], u_kep[3]))

    theta_sp = [k_s,tc_s,a,i,u1_s,u2_s,k1_s] + theta_aux.tolist()
    theta_k2 =  k_k,tc_k,a,i,u1_k,u2_k,k0_k

    ll = spz_loglike(theta_sp, t, f, s, p, aux)
    ll += k2_loglike(theta_k2, k2data[0], k2data[1], k2data[2], p)

    if np.isnan(ll).any():
        return -np.inf
    return ll + lp


def get_theta(theta, sub):

    a,i,k_s,k_k,tc_s,tc_k,u1_s,u2_s,u1_k,u2_k,k1_s,k0_k = theta[:12]
    theta_aux = theta[12:]
    theta_sp = [k_s,tc_s,a,i,u1_s,u2_s,k1_s] + theta_aux.tolist()
    theta_k2 =  k_k,tc_k,a,i,u1_k,u2_k,k0_k

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

    setup ::
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

        if self._tr['i'] > np.pi/2.:
            self._tr['i'] = np.pi - self._tr['i']
        print "\nInitial parameter values:"
        for k,v in self._tr.items():
            if k == 'i':
                print "{} = {}".format(k,v * 180./np.pi)
            else:
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

        method = self._method

        if method == 'cen':

            n = self._xy.shape[0]
            bias = np.repeat(1, n)
            self._aux = np.c_[bias, self._xy].T

        elif method == 'cen-quad':

            n = self._xy.shape[0]
            bias = np.repeat(1, n)
            self._aux = np.c_[bias, self._xy, self._xy**2].T

        elif method == 'pld':

            self._aux = self._pix.T

        elif method == 'pld-quad':

            self._aux = np.c_[self._pix, self._pix**2].T

        elif method == 'base':

            n = self._xy.shape[0]
            bias = np.repeat(1, n)
            self._aux = bias.reshape(1, n)

        elif method == 'pca':

            n = self._xy.shape[0]
            bias = np.repeat(1, n)
            X = self._pix.T
            top2 = util.pca(X, n=2)
            self._aux = np.c_[bias, top2].T

        elif method == 'pca2':

            n = self._xy.shape[0]
            bias = np.repeat(1, n)
            X = np.c_[self._pix, self._pix**2].T
            top2 = util.pca(X, n=2)
            self._aux = np.c_[bias, top2].T

        elif method == 'pca-quad':

            n = self._xy.shape[0]
            bias = np.repeat(1, n)
            X = self._pix.T
            top2 = self._pca(X, n=2)
            self._aux = np.c_[bias, top2, top2**2].T

        else:

            raise ValueError('method must be one of: {}'.format(METHODS))


    def _setup_ld(self):

        teff, uteff = self._setup['stellar']['teff']
        logg, ulogg = self._setup['stellar']['logg']
        feh, ufeh = self._setup['stellar']['feh']

        try:
            self._u_kep = self._setup['ld']['kep']
        except KeyError as e:
            print "Input missing Kepler limb-darkening priors"
            print "Using LDTk..."
            self._u_kep = k2_util.get_ld_ldtk(teff, uteff, logg, ulogg, feh, ufeh)

        try:
            self._u_spz = self._setup['ld']['spz']
        except KeyError as e:
            print "Input missing Spitzer limb-darkening priors"
            print "Using Claret+2012..."
            self._u_spz = util.get_ld_claret(teff, uteff, logg, ulogg, 'S2')

        self._output['ld_priors'] = dict(kep=self._u_kep, spz=self._u_spz)


    def _load_k2(self):

        k2_folded_fp = self._setup['config']['k2lc']

        if os.path.isfile(k2_folded_fp):

            try:

                print "\nLoading K2 data from file: {}".format(k2_folded_fp)
                names = open(k2_folded_fp).readline().split(',')
                if len(names) == 3:
                    self._df_k2 = pd.read_csv(k2_folded_fp, names='t f s'.split())
                else:
                    self._df_k2 = pd.read_csv(k2_folded_fp, names='t f'.split())
                    self._add_k2_sig()

            except:

                raise ValueError('Invalid K2 light curve file format')


    def _load_spz(self):

        print "\nLoading Spitzer data"

        self._radius = self._setup['config']['radius']
        self._aor = self._setup['config']['aor']
        self._data_dir = self._setup['config']['datadir']

        fp = os.path.join(self._data_dir, self._aor+'_phot.pkl')
        df_sp = sxp.util.df_from_pickle(fp, self._radius, pix=True)

        # plot
        fp = os.path.join(self._out_dir, 'spz_raw.png')
        plot.errorbar(df_sp.t, df_sp.f, df_sp.s, fp, alpha=0.5)

        # extract time series and bin
        keys = ['p{}'.format(i) for i in range(9)]
        pix = df_sp[keys].values
        t, f, s = df_sp.t, df_sp.f, df_sp.s
        timestep = np.median(np.diff(t)) * 24 * 3600
        bin_size_sec = self._setup['config']['binsize']
        bs = int(round(bin_size_sec/timestep))
        binned = functools.partial(util.binned, binsize=bs)
        tb, fb, ub, pixb = map(binned, [t, f, s, pix])
        ub /= np.sqrt(bs)
        t, f, s, pix = tb, fb, ub, pixb
        d_sp = {k:v for k,v in zip('t f s'.split(), [t, f, s])}
        self._df_sp = pd.DataFrame(d_sp)
        self._pix = pix

        # plot
        fp = os.path.join(self._out_dir, 'spz_binned.png')
        plot.errorbar(tb, fb, ub, fp)

        cube = pix.reshape(-1,3,3)
        cubestacked = np.median(cube, axis=0)
        fp = os.path.join(self._out_dir, 'spz_pix.png')
        plot.pixels(cubestacked, fp)

        # compute and plot centroids
        cx, cy = centroid_2dg(cubestacked)
        print "Cube centroids: {}, {}".format(cx, cy)
        cx, cy = map(int, map(round, [cx, cy]))

        self._xy = np.array([centroid_com(i) for i in cube])
        x, y = self._xy.T
        fp = os.path.join(self._out_dir, 'spz_cen.png')
        plot.centroids(t, x, y, fp)


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

        return t, f, s, p, aux


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

        return t, f, s, p


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

        return t, f, s, p, aux, k2data, u_kep, u_spz


    @property
    def _ini(self):

        """
        Initial guess parameter vector.
        """

        n_aux = self._aux.shape[0] if self._aux is not None else 0
        a, i, k = self._tr['a'], self._tr['i'], self._tr['k']
        tc_s = self._spz_tc
        tc_k = 0
        u1_s, u2_s = self._u_spz[0], self._u_spz[2]
        u1_k, u2_k = self._u_kep[0], self._u_kep[2]
        k1_s, k0_k = 0, 0

        initial = [a, i, k, k, tc_s, tc_k, u1_s, u2_s,
            u1_k, u2_k, k1_s, k0_k]

        initial += [0] * n_aux

        return np.array(initial)


    @property
    def _labels(self):

        labels = 'a i k_s k_k tc_s tc_k u1_s u2_s u1_k u2_k k1_s k0_k'.split()
        if self._aux is not None:
            labels += ['c{}'.format(i) for i in range(len(self._aux))]

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


    def _map(self, method='nelder-mead'):

        """
        Maximum a posteriori model fit.
        """

        nlp = lambda *x: -self._logprob(*x)
        initial = self._ini
        args = self._args
        res = op.minimize(nlp, initial, args=args, method=method)

        return res


    def max_apo(self, methods=('nelder-mead', 'powell'), plot=True):

        """
        Attempt maximum a posteriori model fit using both Powell and Nelder-Mead.
        """

        print "\nAttempting maximum a posteriori optimization"
        results = []
        for method in methods:
            res = self._map(method=method)
            if res.success:
                print "Log probability ({}): {}".format(method, -res.fun)
                results.append(res)

        if len(results) > 0:
            idx = np.argmin([r.fun for r in results])
            map_best = np.array(results)[idx]
            lp_map = -1 * map_best.fun
            pv_map = map_best.x
            lp_ini = self._logprob(self._ini, *self._args)
            if lp_map > lp_ini:
                self._pv_map = pv_map
                self._lp_map = lp_map
                self._max_apo_alg = np.array(methods)[idx]
                if plot:
                    self._plot_max_apo()
        else:
            print "All methods failed to converge"
            return

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
            fp = os.path.join(self._out_dir, 'fit-map.png')
            fig.savefig(fp)
            pl.close()


    def run_mcmc(self, save=False, restart=False, resume=False):

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

        if self._pv_map is None:
            pv_ini = self._ini
            logprob_ini = self._logprob(pv_ini, *args)
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
                self._post_mcmc()
                return

        ndim = len(pv_ini)
        nwalkers = 8 * ndim if ndim > 12 else 16 * ndim
        print "\nRunning MCMC"
        print "{} walkers exploring {} dimensions".format(nwalkers, ndim)

        sampler = EnsembleSampler(nwalkers, ndim, logprob,
            args=args, threads=nthreads)
        pos0 = sample_ball(pv_ini, [1e-4]*ndim, nwalkers) # FIXME use individual sigmas
        pos0[13] = np.abs(pos0[13])

        print "\nstage 1"
        for pos,_,_ in tqdm(sampler.sample(pos0, iterations=nsteps1)):
            pass

        labels = self._labels
        fp = os.path.join(out_dir, 'chain-initial.png')
        plot.chain(sampler.chain, labels, fp)

        idx = np.argmax(sampler.lnprobability)
        new_best = sampler.flatchain[idx]
        new_prob = sampler.lnprobability.flat[idx]
        best = new_best if new_prob > logprob_ini else pv_ini
        pos = sample_ball(best, [1e-6]*ndim, nwalkers) # FIXME use individual sigmas
        pos[13] = np.abs(pos[13])
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
            msg = "After {} steps\n  Mean G-R: {}\n  Max G-R: {}"
            print msg.format(nsteps, gr.mean(), gr.max())
            if (gr < gr_threshold).all():
                break

        idx = gr_vals[-1] >= gr_threshold
        if idx.any():
            print "WARNING -- some parameters failed to converge below threshold:"
            print self._pv_names(idx)

        fp = os.path.join(out_dir, 'gr.png')
        plot.gr_iter(gr_vals, fp)

        fp = os.path.join(out_dir, 'chain.png')
        plot.chain(sampler.chain, labels, fp)

        self._lp_mcmc = sampler.lnprobability.flatten().max()
        idx = np.argmax(sampler.lnprobability)
        assert sampler.lnprobability.flat[idx] == self._lp_mcmc
        self._pv_mcmc = sampler.flatchain[idx]

        burn = nsteps - nsteps2 if nsteps > nsteps2 else 0
        thin = 1
        self._fc = sampler.chain[:,burn::thin,:].reshape(-1, ndim)
        fp = os.path.join(out_dir, 'corner.png')
        plot.corner(self._fc, labels, fp=fp, truths=self._pv_mcmc)

        if save:
            fp = os.path.join(out_dir, 'mcmc')
            np.savez_compressed(
                fp,
                flat_chain=self._fc,
                logprob_best=self._lp_mcmc,
                pv_best=self._pv_mcmc,
                gelman_rubin=np.array(gr_vals)
                )

        if 'opt' not in self._output.keys():
            self._output['opt'] = {}
        self._output['opt']['mcmc'] = dict(
            logprob=float(self._lp_mcmc),
            pv=dict(zip(self._labels, self._pv_mcmc.tolist()))
            )

        # chain stats
        self._output['stats'] = dict(
            acor=dict(zip(self._labels, sampler.acor.tolist())),
            gr=dict(zip(self._labels, gr_vals[-1].tolist()))
            )

        self._post_mcmc()


    def _post_mcmc(self):

        # Spitzer stats
        best_sp = get_theta(self._pv_mcmc, 'sp')
        mod_full = spz_model(best_sp, *self._spz_args)
        t, f, s = self._spz_ts
        sys = spz_model(best_sp, *self._spz_args, ret_sys=True)
        f_cor = f - sys
        mod_transit = spz_model(best_sp, *self._spz_args, ret_ma=True)
        resid = f - spz_model(best_sp, *self._spz_args)
        fp = os.path.join(self._out_dir, 'fit-mcmc-best.png')
        plot.corrected_ts(t, f, f_cor, mod_full, mod_transit, resid, fp)

        timestep = np.median(np.diff(t)) * 86400
        rms = util.rms(resid)
        beta = util.beta(resid, timestep)
        nd, npar = len(s), len(best_sp)
        rchisq = util.chisq(resid, s, nd, npar, reduced=True)
        obj = lambda x: (1 - util.chisq(resid, s*x, nd, npar, reduced=True))**2
        res = op.minimize(obj, 1, method='nelder-mead')
        if res.success:
            rescale_fac = float(res.x)
        else:
            rescale_fac = None
        bic = util.bic(spz_loglike(best_sp, *self._spz_args), nd, npar)

        print "RMS: {}".format(rms)
        print "Beta: {}".format(beta)
        self._output['spz'] = dict(rms=float(rms),
            beta=float(beta),
            reduced_chisq=float(rchisq),
            rescale_fac=rescale_fac,
            bic=float(bic)
            )

        # update corrected Spitzer data
        if self._lp_mcmc > self._lp_map:
            best = self._pv_mcmc
        else:
            best = self._pv_map
        best_sp = get_theta(best, 'sp')
        self._update_df_sp(best_sp)
        fp = os.path.join(self._out_dir, 'spz.csv')
        self._df_sp.to_csv(fp, index=False)

        # K2 stats
        best_k2 = get_theta(self._pv_mcmc, 'k2')
        df_k2 = self._df_k2
        t, f, s = df_k2['t'], df_k2['f'], df_k2['s']
        model = k2_loglike(best_k2, *self._k2_args, ret_mod=True)
        resid = f - model
        rms = util.rms(resid)
        timestep = np.median(np.diff(t)) * 86400
        try:
            beta = float(util.beta(resid, timestep))
        except:
            beta = None
        nd, npar = len(s), len(best_k2)
        rchisq = util.chisq(resid, s, nd, npar, reduced=True)
        obj = lambda x: (1 - util.chisq(resid, s*x, nd, npar, reduced=True))**2
        res = op.minimize(obj, 1, method='nelder-mead')
        if res.success:
            rescale_fac = float(res.x)
        else:
            rescale_fac = None
        bic = util.bic(k2_loglike(best_k2, *self._k2_args), nd, npar)

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
        rho = util.sample_rhostar(a_samples, p)
        p0 = 1,rho.mean(),rho.std(), 1,np.median(rho),rho.std()
        fp = os.path.join(self._out_dir, 'rhostar.png')
        plot.multi_gauss_fit(rho, p0, fp=fp)

        # small corner
        fp = os.path.join(self._out_dir, 'corner-small.png')
        idx = []
        for n in 'a i k_s k_k tc_s'.split():
            idx += [self._pn_idx(n)]
        fc = self._fc[:,idx].copy()
        tc = int(fc[:,-1].mean())
        fc[:,-1] -= tc
        fc[:,1] *= 180/np.pi
        labels = r'$a/R_{\star}$ $i$ $R_p/R_{\star,S}$ $R_p/R_{\star,K}$'
        labels += r' $T_[C,S]-{}$'.format(tc).replace('[','{').replace(']','}')
        plot.corner(fc, labels.split(), fp=fp,
            quantiles=None, plot_datapoints=False, dpi=256, tight=True)


    def _update_df_sp(self, best_sp):

        args = self._args
        sys = spz_model(best_sp, *self._spz_args, ret_sys=True)
        self._df_sp['f_cor'] = self._df_sp['f'] - sys
        resid = self._df_sp['f'] - spz_model(best_sp, *self._spz_args)
        self._df_sp['resid'] = resid


    def dump(self):

        self._output['ini'] = {}
        for k,v in self._tr.items():
            if k == 'u':
                continue
            self._output['ini'][k] = float(v)
        fp = os.path.join(self._out_dir, 'output.yaml')
        yaml.dump(self._output, open(fp, 'w'), default_flow_style=False)


    def plot_final(self):

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

        percs = [50, 16, 84]
        npercs = len(percs)

        df_k2.ti = np.linspace(df_k2.t.min(), df_k2.t.max(), 1000)
        args_k2 = df_k2.ti, df_k2['f'], df_k2['s'], p

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

        plot.k2_spz_together(self._df_sp, self._df_k2, flux_pc_sp, flux_pc_k2,
            percs, self._fc[:,2], self._fc[:,3], fp, title=self._plot_title)


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
