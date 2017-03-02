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
from emcee import MHSampler, EnsembleSampler, PTSampler
from emcee.utils import sample_ball
import corner
from tqdm import tqdm

from . import plot
from . import util
from . import engines


class Fit(object):

    def __init__(self):

        self._map = None
        self._mcmc = None

    @property
    def _ini(self):

        """
        Returns the initial parameter vector guess.
        """

        raise NotImplementedError

    @property
    def _args(self):

        """
        Returns arguments to be passed to the log-probability function.
        """

        raise NotImplementedError


    def set_ld_prior(self, ldp):

        """
        Sets a quadratic limb darkening prior: [mu1, sig1, mu2, sig2]
        """

        self._ld_prior = ldp


    def run_map(self, methods=('nelder-mead', 'powell'), make_plots=True, nmodel=None):

        """
        Run a maximum a posteriori (MAP) fit using one or more methods.
        Defaults to Nelder-Mead and Powell.
        """

        self._map = engines.MAP(self._logprob, self._ini, self._args, methods=methods)
        self._map.run()
        self._pv_map, self._lp_map, self._max_apo_alg = self._map.results

        if self._pv_map is None:
            self._pv_map = self._ini
            self._lp_map = self._logprob(self._pv_map, *self._args)
            self._max_apo_alg = 'none'

        self._pv_best = self._pv_map
        self._lp_best = self._lp_map

        if make_plots:
            fp = os.path.join(self._out_dir, 'map-bestfit.png')
            self.plot_best(nmodel=nmodel, fp=fp)


    def model(self, t=None, pv=None):
        p = self._p
        if t is None:
            t = self._data[:,0]
        if pv is None:
            pv = self._pv_best
        m = self._logprob(pv, t, *self._args[1:], ret_mod=True)
        return m


    def resid(self, pv):
        t, f = self._data.T
        return f - self.model(pv=pv)


    def plot(self, fp=None, nmodel=None, pv=None, **kwargs):
        t, f = self._data.T
        title = "Std. dev. of residuals: {}".format(np.std(self.resid(pv=pv)))
        if nmodel is not None:
            ti = np.linspace(t.min(), t.max(), nmodel)
            m = self.model(t=ti, pv=pv)
            plot.simple_ts(t, f, tmodel=ti, model=m, fp=fp, title=title, **kwargs)
        else:
            m = self.model(t=t, pv=pv)
            plot.simple_ts(t, f, model=m, fp=fp, title=title, **kwargs)


    @property
    def _pv_names(self):
        return self._logprob(self._ini, *self._args, ret_pvnames=True)


    @property
    def best(self):
        return dict(zip(self._pv_names, self._pv_best))


    def run_mcmc(self, make_plots=True, nmodel=None, restart=False, resume=False, **kwargs):

        ini = self._pv_map
        args = self._args
        names = self._pv_names

        if self._out_dir is not None:

            fp = os.path.join(self._out_dir, 'mcmc.npz')
            if os.path.isfile(fp):

                if resume:

                    print "Resuming from previous best position"
                    npz = np.load(fp)
                    ini = npz['pv_best']

                elif not restart:

                    print "Loading chain from previous run"
                    npz = np.load(fp)
                    self._pv_mcmc = npz['pv_best']
                    self._lp_mcmc = npz['logprob_best']
                    self._fc = npz['flat_chain']
                    self._gr = npz['gelman_rubin']

                    if self._lp_mcmc > self._lp_best:
                        self._pv_best = self._pv_mcmc
                        self._lp_best = self._lp_mcmc

                    return

        self._mcmc = engines.MCMC(self._logprob, ini, args, names, outdir=self._out_dir)
        self._mcmc.run(make_plots=make_plots, **kwargs)
        pv, lp, fc, gr, acor = self._mcmc.results

        self._pv_mcmc = pv
        self._lp_mcmc = lp
        self._fc = fc
        self._gr = gr
        self._acor = acor

        if self._lp_mcmc > self._lp_best:
            self._pv_best = self._pv_mcmc
            self._lp_best = self._lp_mcmc

        if make_plots:

            fp = os.path.join(self._out_dir, 'mcmc-bestfit.png')
            self.plot_best(nmodel=nmodel, fp=fp)

            fp = os.path.join(self._out_dir, 'mcmc-samples.png')
            self.plot_samples(nmodel=nmodel, fp=fp)


    def plot_best(self, nmodel=None, fp=None):

            assert self._map._hasrun or self._mcmc._hasrun

            t, f = self._data.T

            if nmodel is not None:
                ti = np.linspace(t.min(), t.max(), nmodel)
            else:
                ti = t

            m = self.model(t=ti)
            plot.simple_ts(t, f, tmodel=ti, model=m, fp=fp)


    def plot_samples(self, nmodel=None, fp=None):

            assert self._mcmc._hasrun

            t, f = self._data.T

            if nmodel is not None:
                ti = np.linspace(t.min(), t.max(), nmodel)
            else:
                ti = t

            fc = self._fc
            ps = [self.model(t=ti, pv=s) for s in fc[np.random.randint(len(fc), size=100)]]
            plot.samples(t, f, ps, tmodel=ti, fp=fp)


    def summarize_mcmc(self, save=True):

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

        if save:
            fp = os.path.join(self._out_dir, 'mcmc-summary.yaml')
            yaml.dump(summary, open(fp, 'w'), default_flow_style=False)
        else:
            return summary
