from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
import scipy.optimize as op
from emcee import MHSampler, EnsembleSampler
from emcee.utils import sample_ball
from tqdm import tqdm

from . import util
from . import plot



class Engine(object):

    def __init__(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    @property
    def results(self):
        raise NotImplementedError



class MAP(Engine):

    def __init__(self, logprob, ini, args, methods=('nelder-mead', 'powell')):

        """
        Maximum a posteriori model fit. Defaults to Nelder-Mead and Powell.

        :param logprob  : log-probability (posterior) function to maximize
        :param ini      : initial parameter vector
        :param args     : any additional args to be passed to logprob
        :param methods  : list of methods to be used from scipy.optimize
        """

        self._logprob = logprob
        self._ini = ini
        self._args = args
        self._methods = methods
        self._pv = None
        self._lp = None
        self._method = None
        self._hasrun = False


    def _map(self, method='nelder-mead'):

        nlp = lambda *x: -self._logprob(*x)
        initial = self._ini
        args = self._args
        res = op.minimize(nlp, initial, args=args, method=method)

        return res


    def run(self):

        print("\nAttempting maximum a posteriori optimization")
        results = []
        for method in self._methods:
            res = self._map(method=method)
            if res.success:
                print("Log probability ({}): {}".format(method, -res.fun))
                results.append(res)

        if len(results) > 0:
            idx = np.argmin([r.fun for r in results])
            map_best = np.array(results)[idx]
            lp_map = -1 * map_best.fun
            pv_map = map_best.x
            lp_ini = self._logprob(self._ini, *self._args)
            if lp_map > lp_ini:
                method = np.array(self._methods)[idx]
                self._pv, self._lp, self._method = pv_map, lp_map, method
            else:
                self._pv, self._lp, self._method = self._ini, lp_ini, 'initial'
        else:
            print("All methods failed to converge")

        self._hasrun = True


    @property
    def results(self):

        if not self._hasrun:
            print("Need to call run() first!")

        return self._pv, self._lp, self._method



class MCMC(Engine):

    def __init__(self, logprob, ini, args, names, outdir=None):

        """
        Affine invariant ensemble MCMC exploration of parameter space.

        :param logprob  : log-probability (posterior) function to maximize
        :param ini      : initial parameter vector
        :param args     : any additional args to be passed to logprob
        :param names    : list of parameter names
        :param outdir   : path to output directory
        """

        self._logprob = logprob
        self._ini = ini
        self._args = args
        self._names = names
        self._outdir = outdir
        self._logprob_ini = logprob(ini, *args)
        self._hasrun = False
        self._ndim = len(ini)


    def run(self, nproc=1, nsteps1=1e3, nsteps2=1e3, max_steps=1e4, burn=None, gr_threshold=1.1, save=True, make_plots=True, restart=False, resume=False):

        """
        :param nproc        : number of processes to use for sampling
        :param nsteps1      : number of steps to take during stage 1 exploration
        :param nsteps2      : number of steps to take during each stage 2 iteration
        :param max_steps    : maximum number of steps to take during stage 2
        :param gr_threshold : Gelman-Rubin convergence threshold
        :param save         : whether to save MCMC samples and related output
        :param plot         : whether to generate plots
        :param restart      : whether to restart (if previous run exists)
        :param resume       : whether to resume (if previous run exists)
        """


        if save or make_plots:
            assert self._outdir is not None

        FILE_EXISTS = False
        if self._outdir is not None:
            fp = os.path.join(self._outdir, 'mcmc.npz')
            FILE_EXISTS = os.path.isfile(fp)

        if FILE_EXISTS and resume:

            print("Resuming from previous best position")
            npz = np.load(fp)
            pv_ini = npz['pv_best']
            lp_ini = npz['logprob_best']
            self._run(pv_ini, lp_ini, nproc=nproc, nsteps1=nsteps1, nsteps2=nsteps2, max_steps=max_steps, gr_threshold=gr_threshold, make_plots=make_plots)
            if burn is None:
                burn = nsteps2 if self._nsteps > nsteps2 else 0
            self._burn_thin(burn=burn, make_plots=make_plots)
            if save:
                self._save()

        elif FILE_EXISTS and not restart:

            print("Loading chain from previous run")
            npz = np.load(fp)
            self._pv = npz['pv_best']
            self._lp_max = npz['logprob_best']
            self._fc = npz['flat_chain']
            self._lp_flat = npz['logprob_flat']
            self._gr = npz['gelman_rubin']
            self._af = npz['acceptance_fraction']
            self._c = npz['chain']
            self._lp = npz['logprob']
            self._burn_thin(burn=burn, make_plots=make_plots)
            self._nwalkers, self._nsteps, self._ndim = self._c.shape
            self._hasrun = True

        elif not FILE_EXISTS or restart:

            pv_ini = self._ini
            lp_ini = self._logprob_ini
            self._run(pv_ini, lp_ini, nproc=nproc, nsteps1=nsteps1, nsteps2=nsteps2, max_steps=max_steps, gr_threshold=gr_threshold, make_plots=make_plots)
            if burn is None:
                burn = nsteps2 if self._nsteps > nsteps2 else 0
            self._burn_thin(burn=burn, make_plots=make_plots)
            if save:
                self._save()


    def _run(self, pv_ini, lp_ini, nproc=4, nsteps1=1e3, nsteps2=1e3, max_steps=1e4, gr_threshold=1.1, make_plots=True):

        logprob = self._logprob
        args = self._args
        names = self._names
        logprob_ini = lp_ini
        ndim = self._ndim
        nwalkers = 8 * ndim if ndim > 12 else 16 * ndim

        print("\nRunning MCMC")
        print("{} walkers exploring {} dimensions".format(nwalkers, ndim))

        sampler = EnsembleSampler(nwalkers, ndim, logprob,
            args=args, threads=nproc)
        pos0 = sample_ball(pv_ini, [1e-4]*ndim, nwalkers)

        print("\nstage 1")
        for pos,_,_ in tqdm(sampler.sample(pos0, iterations=nsteps1)):
            pass

        if make_plots:
            fp = os.path.join(self._outdir, 'mcmc-chain-initial.png')
            plot.chain(sampler.chain, names, burn=None, fp=fp)

        idx = np.argmax(sampler.lnprobability)
        new_best = sampler.flatchain[idx]
        new_prob = sampler.lnprobability.flat[idx]
        best = new_best if new_prob > logprob_ini else pv_ini
        pos = sample_ball(best, [1e-6]*ndim, nwalkers)
        sampler.reset()

        print("\nstage 2")
        nsteps = 0
        gr_vals = []
        while nsteps < max_steps:
            for pos,_,_ in tqdm(sampler.sample(pos, iterations=nsteps2)):
                pass
            nsteps += nsteps2
            gr = util.stats.gelman_rubin(sampler.chain)
            gr_vals.append(gr)
            msg = "After {} steps\n  Mean G-R: {}\n  Max G-R: {}"
            print(msg.format(nsteps, gr.mean(), gr.max()))
            if (gr < gr_threshold).all():
                break

        idx = gr_vals[-1] >= gr_threshold
        if idx.any():
            print("WARNING -- some parameters failed to converge below threshold:")
            print(np.array(names)[idx])

        self._c = sampler.chain
        self._lp = sampler.lnprobability
        self._lp_max = sampler.lnprobability.flatten().max()
        idx = np.argmax(sampler.lnprobability)
        assert sampler.lnprobability.flat[idx] == self._lp_max
        self._pv = sampler.flatchain[idx]
        self._gr = np.array(gr_vals)
        self._af = sampler.acceptance_fraction
        self._hasrun = True
        self._nsteps = nsteps
        self._nwalkers = nwalkers


    def _burn_thin(self, burn=None, thin=10, make_plots=True):

        self._fc = self._c[:,burn::thin,:].reshape(-1, self._ndim)
        self._lp_flat = self._lp[:,burn::thin].reshape(-1)

        if make_plots:

            print("Making MCMC plots")
            assert self._outdir is not None

            fp = os.path.join(self._outdir, 'mcmc-gr.png')
            plot.gr_iter(self._gr, fp=fp)

            fp = os.path.join(self._outdir, 'mcmc-chain-final.png')
            plot.chain(self._c, self._names, burn=burn, fp=fp)

            fp = os.path.join(self._outdir, 'mcmc-corner.png')
            plot.corner(self._fc, self._names, fp=fp, truths=self._pv)


    def _save(self):
        fp = os.path.join(self._outdir, 'mcmc')
        np.savez_compressed(
            fp,
            ndim=self._ndim,
            nwalkers=self._nwalkers,
            nsteps=self._nsteps,
            chain=self._c,
            logprob=self._lp,
            flat_chain=self._fc,
            logprob_flat=self._lp_flat,
            pv_best=self._pv,
            logprob_best=self._lp_max,
            pv_names=self._names,
            acceptance_fraction=self._af,
            gelman_rubin=self._gr
            )


    @property
    def results(self):

        if not self._hasrun:
            print("Need to call run() first!")

        return self._pv, self._lp_max, self._fc, self._gr, self._af, self._c
