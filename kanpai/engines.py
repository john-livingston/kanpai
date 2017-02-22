import os
import numpy as np
import scipy.optimize as op
from emcee import MHSampler, EnsembleSampler
from emcee.utils import sample_ball
from tqdm import tqdm

import util
import plot



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

    def _map(self, method='nelder-mead'):

        nlp = lambda *x: -self._logprob(*x)
        initial = self._ini
        args = self._args
        res = op.minimize(nlp, initial, args=args, method=method)

        return res

    def run(self):

        print "\nAttempting maximum a posteriori optimization"
        results = []
        for method in self._methods:
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
                method = np.array(self._methods)[idx]
                self._pv, self._lp, self._method = pv_map, lp_map, method
            else:
                self._pv, self._lp, self._method = self._ini, lp_ini, 'initial'
        else:
            print "All methods failed to converge"

        self._hasrun = True

    @property
    def results(self):

        if not self._hasrun:
            print "Need to call run() first!"

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

    def run(self, nproc=4, nsteps1=1e3, nsteps2=1e3, max_steps=1e4, gr_threshold=1.1, pos_idx=None, save=True, make_plots=True):

        """
        :param nproc        : number of processes to use for sampling
        :param nsteps1      : number of steps to take during stage 1 exploration
        :param nsteps2      : number of steps to take during each stage 2 iteration
        :param max_steps    : maximum number of steps to take during stage 2
        :param gr_threshold : Gelman-Rubin convergence threshold
        :param pos_idx      : indices of any parameters that should always be positive
        :param save         : whether or not to save MCMC samples and related output
        :param plot         : whether or not to generate plots
        """

        logprob = self._logprob
        pv_ini = self._ini
        args = self._args
        names = self._names
        logprob_ini = self._logprob_ini

        if save or make_plots:
            assert self._outdir is not None

        ndim = len(pv_ini)
        nwalkers = 8 * ndim if ndim > 12 else 16 * ndim
        print "\nRunning MCMC"
        print "{} walkers exploring {} dimensions".format(nwalkers, ndim)

        sampler = EnsembleSampler(nwalkers, ndim, logprob,
            args=args, threads=nproc)
        pos0 = sample_ball(pv_ini, [1e-4]*ndim, nwalkers) # FIXME use individual sigmas
        if pos_idx is not None:
            pos0[pos_idx] = np.abs(pos0[pos_idx])

        print "\nstage 1"
        for pos,_,_ in tqdm(sampler.sample(pos0, iterations=nsteps1)):
            pass

        if make_plots:
            fp = os.path.join(self._outdir, 'mcmc-chain-initial.png')
            plot.chain(sampler.chain, names, fp)

        idx = np.argmax(sampler.lnprobability)
        new_best = sampler.flatchain[idx]
        new_prob = sampler.lnprobability.flat[idx]
        best = new_best if new_prob > logprob_ini else pv_ini
        pos = sample_ball(best, [1e-6]*ndim, nwalkers) # FIXME use individual sigmas
        if pos_idx is not None:
            pos0[pos_idx] = np.abs(pos0[pos_idx])
        sampler.reset()

        print "\nstage 2"
        nsteps = 0
        gr_vals = []
        while nsteps < max_steps:
            for pos,_,_ in tqdm(sampler.sample(pos, iterations=nsteps2)):
                pass
            nsteps += nsteps2
            gr = util.stats.gelman_rubin(sampler.chain)
            gr_vals.append(gr)
            msg = "After {} steps\n  Mean G-R: {}\n  Max G-R: {}"
            print msg.format(nsteps, gr.mean(), gr.max())
            if (gr < gr_threshold).all():
                break

        idx = gr_vals[-1] >= gr_threshold
        if idx.any():
            print "WARNING -- some parameters failed to converge below threshold:"
            print np.array(names)[idx]

        self._lp = sampler.lnprobability.flatten().max()
        idx = np.argmax(sampler.lnprobability)
        assert sampler.lnprobability.flat[idx] == self._lp
        self._pv = sampler.flatchain[idx]

        burn = nsteps - nsteps2 if nsteps > nsteps2 else 0
        thin = 1
        self._fc = sampler.chain[:,burn::thin,:].reshape(-1, ndim)
        self._lps = sampler.lnprobability[:,burn::thin].reshape(-1)

        if make_plots:
            fp = os.path.join(self._outdir, 'mcmc-gr.png')
            plot.gr_iter(gr_vals, fp)

            fp = os.path.join(self._outdir, 'mcmc-chain-final.png')
            plot.chain(sampler.chain, names, fp)

            fp = os.path.join(self._outdir, 'mcmc-corner.png')
            plot.corner(self._fc, names, fp=fp, truths=self._pv)

        if save:
            fp = os.path.join(self._outdir, 'mcmc')
            np.savez_compressed(
                fp,
                flat_chain=self._fc,
                logprob=self._lps,
                logprob_best=self._lp,
                pv_best=self._pv,
                pv_names=names,
                gelman_rubin=np.array(gr_vals)
                )

        self._gr = gr_vals[-1]
        try:
            self._acor = sampler.acor
        except:
            self._acor = None
        self._hasrun = True

    @property
    def results(self):

        if not self._hasrun:
            print "Need to call run() first!"

        return self._pv, self._lp, self._fc, self._gr, self._acor
