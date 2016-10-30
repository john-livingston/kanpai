#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

import os
import sys
import yaml

import numpy as np
np.warnings.simplefilter('ignore')

import pandas as pd
import scipy.optimize as op
from scipy import stats

from photutils.morphology import centroid_com, centroid_2dg

import seaborn as sb

from emcee import MHSampler, EnsembleSampler, PTSampler
from emcee.utils import sample_ball
import corner

import pickle
import functools

from tqdm import tqdm

import sxp

from .. import util


from pytransit import MandelAgol
MA_K2 = MandelAgol(supersampling=8, exptime=0.02)
MA_SP = MandelAgol()


def model(theta, t, f, s, p, aux, ret_ma=False, ret_sys=False):
    ks,kk,tc,a,i,u1s,u2s,u1k,u2k,t0,sig,k0,k1 = theta[:13]
    auxcoeff = theta[13:]
    ma = MA_SP.evaluate(t, ks, [u1s, u2s], tc, p, a, i)
    bl = k0 + k1 * (t-t.mean())
    if aux.shape[0] == aux.size:
        sys = auxcoeff * aux
    else:
        sys = (auxcoeff * aux.T).sum(axis=1)
    if ret_ma:
        return ma
    if ret_sys:
        return bl + sys
    return ma + bl + sys


def loglike1(theta, t, f, s, p, aux):
    m = model(theta, t, f, s, p, aux)
    resid = f - m
    inv_sigma2 = 1.0/(s**2)
    return -0.5*(np.sum((resid)**2*inv_sigma2 - np.log(inv_sigma2)))


def loglike2(theta, k2, p, ret_ma=False):
    t, f = k2[0], k2[1]
    ks,kk,tc,a,i,u1s,u2s,u1k,u2k,t0,sig,k0,k1 = theta[:13]
    m = MA_K2.evaluate(t, kk, [u1k, u2k], t0, p, a, i)
    if ret_ma:
        return m
    resid = f - m
    inv_sigma2 = 1.0/(sig**2)
    return -0.5*(np.sum((resid)**2*inv_sigma2 - np.log(inv_sigma2)))


def loglike(theta, t, f, s, p, aux, k2):
    return loglike1(theta, t, f, s, p, aux) + loglike2(theta, k2, p)


def logprob(theta, t, f, s, p, aux, k2, u_kep, u_spz):

    ks,kk,tc,a,i,u1s,u2s,u1k,u2k,t0,sig,k0,k1 = theta[:13]

    if ks < 0 or kk < 0 or tc < t.min() or tc > t.max() or i > np.pi/2:
        return -np.inf
    lp = np.log(stats.norm.pdf(u1s, u_spz[0], u_spz[1]))
    lp += np.log(stats.norm.pdf(u2s, u_spz[2], u_spz[3]))
    lp += np.log(stats.norm.pdf(u1k, u_kep[0], u_kep[1]))
    lp += np.log(stats.norm.pdf(u2k, u_kep[2], u_kep[3]))

    ll = loglike(theta, t, f, s, p, aux, k2)

    if np.isnan(ll).any():
        return -np.inf
    return ll + lp



def go(setup, method, bin_size, nsteps1, nsteps2, max_steps,
       data_dir, out_dir, save, nthreads, k2_kolded_fp, restart):

    tr = setup['transit']
    if tr['i'] > np.pi/2.:
        tr['i'] = np.pi - tr['i']
    for k,v in tr.items():
        print "{} = {}".format(k,v)

    teff, uteff = setup['stellar']['teff']
    logg, ulogg = setup['stellar']['logg']
    u_kep, u_spz = util.get_ld(teff, uteff, logg, ulogg)

    radius = setup['config']['radius']
    aor = setup['config']['aor']
    fp = os.path.join(data_dir, aor+'_phot.pkl')
    spz = sxp.util.df_from_pickle(fp, radius, pix=True)

    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(15,5))
        ax.errorbar(spz.t, spz.f, spz.s, alpha=0.5)
        pl.setp(ax, xlim=[spz.t.min(), spz.t.max()])
        fp = os.path.join(out_dir, 'spz_raw.png')
        fig.savefig(fp)
        pl.close()

    keys = ['p{}'.format(i) for i in range(9)]
    pix = spz[keys].values
    t, f, s = spz.t, spz.f, spz.s

    timestep = np.median(np.diff(t)) * 24 * 3600
    bs_sec = bin_size
    bs = int(round(bs_sec/timestep))
    binned = functools.partial(util.binned, binsize=bs)
    tb, fb, ub, pixb = map(binned, [t, f, s, pix])
    ub /= np.sqrt(bs)

    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(15,3), sharex=True, sharey=True)
        ax.errorbar(tb, fb, ub, marker='o', linestyle='none')
        pl.setp(ax, xlim=[tb.min(), tb.max()])
        fp = os.path.join(out_dir, 'spz_binned.png')
        fig.savefig(fp)
        pl.close()

    t, f, s, pix = tb, fb, ub, pixb

    cube = pix.reshape(-1,3,3)
    cubestacked = np.median(cube, axis=0)
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(5,5))
        pl.imshow(cubestacked, interpolation='none')
        fp = os.path.join(out_dir, 'spz_pix.png')
        fig.savefig(fp)
        pl.close()

    cx, cy = centroid_2dg(cubestacked)
    print "cube centroids: {}, {}".format(cx, cy)
    cx, cy = map(int, map(round, [cx, cy]))
    # print cx, cy

    xy = np.array([centroid_com(i) for i in cube])
    x, y = xy.T
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(15,5))
        ax.plot(t, x, label='x')
        ax.plot(t, y, label='y')
        ax.legend()
        fp = os.path.join(out_dir, 'spz_cen.png')
        pl.setp(ax, xlim=[t.min(), t.max()])
        fig.savefig(fp)
        pl.close()

    names = open(k2_kolded_fp).readline().split()
    if len(names) == 3:
        df = pd.read_csv(k2_kolded_fp, sep=' ', names='t f s'.split())
    else:
        df = pd.read_csv(k2_kolded_fp, sep=' ', names='t f'.split())
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(15,5))
        ax.plot(df.t, df.f, marker='o', lw=0)
        fp = os.path.join(out_dir, 'k2_folded.png')
        fig.savefig(fp)
        pl.close()

    if method == 'cen':
        aux = np.c_[x, y].T
    elif method == 'pld':
        aux = pix.T
    else:
        sys.exit('neither cen nor pld selected')

    p = tr['p']
    k2 = df[['t','f']].values.T
    args = (t, f, s, p, aux, k2, u_kep, u_spz)
    initial = [tr['k'], tr['k'], t.mean(), tr['a'], tr['i'],
               u_spz[0], u_spz[2], u_kep[0], u_kep[2], 0, 1e-5, 0, 0] + [0] * aux.shape[0]

    # nlp = lambda *x: -logprob(*x)
    # res = op.minimize(nlp, initial, args=args, method='nelder-mead')
    # if res.success:
    #     print res.x
    #
    # with sb.axes_style(axes_style):
    #     fig, axs = pl.subplots(1, 2, figsize=(15,3), sharex=True, sharey=True)
    #     axs[0].plot(t, f, 'k.')
    #     axs[0].plot(t, model(initial, *args[:-3]), 'b-', lw=5)
    #     axs[0].plot(t, model(res.x, *args[:-3]), 'r-', lw=5)
    #     axs[1].plot(t, f-model(res.x, *args[:-3], ret_sys=True), 'k.')
    #     axs[1].plot(t, model(initial, *args[:-3], ret_ma=True), 'b-', lw=5)
    #     axs[1].plot(t, model(res.x, *args[:-3], ret_ma=True), 'r-', lw=5)
    #     pl.setp(axs, xlim=[t.min(), t.max()], xticks=[], yticks=[])
    #     fig.tight_layout()
    #     fp = os.path.join(out_dir, 'fit-map.png')
    #     fig.savefig(fp)
    #     pl.close()


    ndim = len(initial)
    nwalkers = 8*ndim

    fp = os.path.join(out_dir, 'flatchain.npz')
    if os.path.isfile(fp) and not restart:

        print "using chain from previous run"
        npz = np.load(fp)
        fc = npz['fc']
        best = npz['best']

    else:

        sampler = EnsembleSampler(nwalkers, ndim, logprob, args=args, threads=nthreads)
        pos0 = sample_ball(initial, [1e-3]*ndim, nwalkers)

        width = 30
        print "\nstage 1"
        for pos,_,_ in tqdm(sampler.sample(pos0, iterations=nsteps1)):
            pass

        idx = np.argmax(sampler.lnprobability)
        best = sampler.flatchain[idx]
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
            gr_vals.append(gr.mean())
            msg = "After {} steps\n\tMean G-R: {}\n\tMax G-R: {}"
            print msg.format(nsteps, gr.mean(), gr.max())
            if (gr < 1.1).all():
                break
            # worst = np.argmax(gr)
            # best = np.argmin(gr)
            # pos[worst] = pos[best]

        with sb.axes_style('white'):
            fig, ax = pl.subplots(1, 1, figsize=(5,2))
            ax.plot(gr_vals, 'k-')
            pl.setp(ax, xlabel='iterations', ylabel='mean G-R')
            fp = os.path.join(out_dir, 'gr.png')
            fig.savefig(fp)
            pl.close()

        chain = sampler.chain
        labels = 'ks,kk,tc,a,i,u1s,u2s,u1k,u2k,t0,sig,k0,k1'.split(',') + ['c{}'.format(i) for i in range(len(aux))]
        with sb.axes_style('white'):
            fig, axs = pl.subplots(ndim, 1, figsize=(15,ndim/1.5), sharex=True)
            [axs.flat[i].plot(c, drawstyle='steps', color='k', alpha=4./nwalkers) for i,c in enumerate(chain.T)]
            [pl.setp(axs.flat[i], ylabel=labels[i]) for i,c in enumerate(chain.T)]
            fp = os.path.join(out_dir, 'chains.png')
            fig.savefig(fp)
            pl.close()


        burn = nsteps - 1000 if nsteps > 2000 else 0
        thin = 1
        fc = chain[:,burn::thin,:].reshape(-1, ndim)

        hist_kwargs = dict(lw=2, alpha=0.5)
        title_kwargs = dict(fontdict=dict(fontsize=12))
        with sb.axes_style('white'):
            corner.corner(fc,
                          labels=labels,
                          hist_kwargs=hist_kwargs,
                          title_kwargs=title_kwargs,
                          show_titles=True,
                          quantiles=[0.16,0.5,0.84],
                          title_fmt='.4f')
            fp = os.path.join(out_dir, 'corner.png')
            pl.savefig(fp)
            pl.close()


        maxprob = sampler.lnprobability.flatten().max()
        # print maxprob
        idx = np.argmax(sampler.lnprobability)
        assert sampler.lnprobability.flat[idx] == maxprob
        best = sampler.flatchain[idx]
        # print best

        with sb.axes_style('white'):
            fig, axs = pl.subplots(1, 3, figsize=(11,3), sharex=True, sharey=False)
            axs.flat[0].plot(t, f, 'k.')
            axs.flat[0].plot(t, model(best, *args[:-3]), '-', lw=2)
            axs.flat[1].plot(t, f - model(best, *args[:-3], ret_sys=True), 'k.')
            axs.flat[1].plot(t, model(best, *args[:-3], ret_ma=True), '-', lw=5)
            resid = f - model(best, *args[:-3])
            axs.flat[2].plot(t, resid, 'k.')
            pl.setp(axs, xlim=[t.min(), t.max()], xticks=[], yticks=[])
            fig.tight_layout()
            fp = os.path.join(out_dir, 'fit-mcmc-best.png')
            fig.savefig(fp)
            pl.close()

        timestep = np.median(np.diff(t))*24*3600
        rms = util.rms(resid)
        beta = util.beta(resid, timestep)
        print "RMS: {}".format(rms)
        print "Beta: {}".format(beta)
        fp = os.path.join(out_dir, 'stats.txt')
        with open(fp, 'w') as o:
            o.write("Method: {}\n".format(method))
            o.write("RMS: {}\n".format(rms))
            o.write("Beta: {}\n".format(beta))


        if save:
            fp = os.path.join(out_dir, 'flatchain')
            np.savez_compressed(fp, fc=fc, best=best)


    tc = np.median(fc[:,2])
    spz_phase = list(t - tc)

    alpha = 0.8
    sb.palettes.set_color_codes(palette='muted')
    with sb.axes_style('whitegrid'):
        fig, axs = pl.subplots(1, 3, figsize=(11,3), sharex=False, sharey=False)

        # percentiles = [50, 0.15, 99.85, 2.5, 97.5, 16, 84]
        # percentiles = [50, 2.5, 97.5, 16, 84]
        percentiles = [50, 16, 84]
        npercs = len(percentiles)

        flux_pr = []
        for theta in fc[np.random.permutation(fc.shape[0])[:1000]]:
            flux_pr.append(loglike2(theta, k2, p, ret_ma=True))
        flux_pr = np.array(flux_pr)
        flux_pc = np.array(np.percentile(flux_pr, percentiles, axis=0))
        axs.flat[0].plot(df.t, df.f, 'k.', alpha=alpha)
        [axs.flat[0].fill_between(df.t, *flux_pc[i:i+2,:], alpha=0.4,
            facecolor='b', edgecolor='b') for i in range(1,npercs-1,2)]
        # axs.flat[0].plot(df.t, loglike2(best, k2, p, ret_ma=True), 'b-', lw=2)
        axs.flat[0].plot(df.t, flux_pc[0], 'b-', lw=1.5)

        flux_pr = []
        for theta in fc[np.random.permutation(fc.shape[0])[:1000]]:
            flux_pr.append(model(theta, *args[:-3], ret_ma=True))
        flux_pr = np.array(flux_pr)
        flux_pc = np.array(np.percentile(flux_pr, percentiles, axis=0))
        fcor = f - model(best, *args[:-3], ret_sys=True)
        # fcor = f - model(np.median(fc, axis=0), *args[:-3], ret_sys=True)
        axs.flat[1].plot(t, fcor, 'k.', alpha=alpha)
        [axs.flat[1].fill_between(t, *flux_pc[i:i+2,:], alpha=0.4,
            facecolor='r', edgecolor='r') for i in range(1,npercs-1,2)]
        # axs.flat[1].plot(t, model(best, *args[:-3], ret_ma=True), 'r-', lw=2)
        axs.flat[1].plot(t, flux_pc[0], 'r-', lw=1.5)

        edgs, bins = np.histogram(fc[:,:2], bins=30)
        axs.flat[2].hist(fc[:,1], bins=bins, histtype='stepfilled', color='b', alpha=alpha, lw=0, label='K2')
        axs.flat[2].hist(fc[:,0], bins=bins, histtype='stepfilled', color='r', alpha=alpha, lw=0, label='Spitzer')
        # axs.flat[2].legend(loc=2)

        ylim = axs.flat[1].get_ylim()
        pl.setp(axs.flat[0], xlim=[spz_phase[0], spz_phase[-1]], ylim=ylim, xticks=[], yticks=[])
        pl.setp(axs.flat[1], xlim=[t.min(), t.max()], ylim=ylim, xticks=[], yticks=[])
        pl.setp(axs.flat[2], xticks=[], yticks=[])

        fig.tight_layout()
        fp = os.path.join(out_dir, 'fit-final.png')
        fig.savefig(fp, dpi=256)
        pl.close()
