
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

import util
import k2
import spz
import plot

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

    ks,kk,tc,a,i,u1s,u2s,u1k,u2k,t0,sig,k0,k1 = theta[:13]
    theta_aux = theta[13:]

    if ks < 0 or kk < 0 or tc < t.min() or tc > t.max() or i > np.pi/2:
        return -np.inf
    lp = np.log(stats.norm.pdf(u1s, u_spz[0], u_spz[1]))
    lp += np.log(stats.norm.pdf(u2s, u_spz[2], u_spz[3]))
    lp += np.log(stats.norm.pdf(u1k, u_kep[0], u_kep[1]))
    lp += np.log(stats.norm.pdf(u2k, u_kep[2], u_kep[3]))

    theta_sp = [ks,tc,a,i,u1s,u2s,k0,k1] + theta_aux.tolist()
    theta_k2 = kk,t0,a,i,u1k,u2k,sig

    ll = spz.loglike(theta_sp, t, f, s, p, aux)
    ll += k2.loglike(theta_k2, k2data[0], k2data[1], p)

    if np.isnan(ll).any():
        return -np.inf
    return ll + lp


def get_theta(theta, sub):

    ks,kk,tc,a,i,u1s,u2s,u1k,u2k,t0,sig,k0,k1 = theta[:13]
    theta_aux = theta[13:]
    theta_sp = [ks,tc,a,i,u1s,u2s,k0,k1] + theta_aux.tolist()
    theta_k2 = kk,t0,a,i,u1k,u2k,sig

    if sub == 'sp':
        return theta_sp
    elif sub == 'k2':
        return theta_k2


def go(setup, method, bin_size, nsteps1, nsteps2, max_steps,
    gr_threshold, out_dir, save, nthreads, k2_kolded_fp, restart):

    fp = os.path.join(out_dir, 'input.yaml')
    yaml.dump(setup, open(fp, 'w'))

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
    data_dir = setup['config']['data_dir']
    fp = os.path.join(data_dir, aor+'_phot.pkl')
    df_sp = sxp.util.df_from_pickle(fp, radius, pix=True)


    fp = os.path.join(out_dir, 'spz_raw.png')
    plot.errorbar(df_sp.t, df_sp.f, df_sp.s, fp, alpha=0.5)


    keys = ['p{}'.format(i) for i in range(9)]
    pix = df_sp[keys].values
    t, f, s = df_sp.t, df_sp.f, df_sp.s


    timestep = np.median(np.diff(t)) * 24 * 3600
    bs_sec = bin_size
    bs = int(round(bs_sec/timestep))
    binned = functools.partial(util.binned, binsize=bs)
    tb, fb, ub, pixb = map(binned, [t, f, s, pix])
    ub /= np.sqrt(bs)

    fp = os.path.join(out_dir, 'spz_binned.png')
    plot.errorbar(tb, fb, ub, fp)

    t, f, s, pix = tb, fb, ub, pixb
    d_sp = {k:v for k,v in zip('t f s'.split(), [t, f, s])}
    df_sp = pd.DataFrame(d_sp)


    cube = pix.reshape(-1,3,3)
    cubestacked = np.median(cube, axis=0)
    fp = os.path.join(out_dir, 'spz_pix.png')
    plot.pixels(cubestacked, fp)


    cx, cy = centroid_2dg(cubestacked)
    print "cube centroids: {}, {}".format(cx, cy)
    cx, cy = map(int, map(round, [cx, cy]))


    xy = np.array([centroid_com(i) for i in cube])
    x, y = xy.T
    fp = os.path.join(out_dir, 'spz_cen.png')
    plot.centroids(t, x, y, fp)


    names = open(k2_kolded_fp).readline().split(',')
    if len(names) == 3:
        df_k2 = pd.read_csv(k2_kolded_fp, names='t f s'.split())
    else:
        df_k2 = pd.read_csv(k2_kolded_fp, names='t f'.split())
    fp = os.path.join(out_dir, 'k2_folded.png')
    plot.simple_ts(df_k2.t, df_k2.f, fp, color='b')


    if method == 'cen':
        aux = np.c_[x, y].T
    elif method == 'pld':
        aux = pix.T
    else:
        sys.exit('neither cen nor pld selected')


    p = tr['p']
    k2data = df_k2[['t','f']].values.T
    args = (t, f, s, p, aux, k2data, u_kep, u_spz)
    initial = np.array( [tr['k'], tr['k'], t.mean(), tr['a'], tr['i'],
               u_spz[0], u_spz[2], u_kep[0], u_kep[2], 0, 1e-5, 0, 0] + [0] * aux.shape[0] )


    nlp = lambda *x: -logprob(*x)
    algs = ['powell', 'nelder-mead']
    best = np.inf
    results = []
    for alg in algs:
        print "attempting minimization with {}".format(alg)
        res = op.minimize(nlp, initial, args=args, method=alg)
        if res.success:
            print "{} negative log probability: {}".format(alg, res.fun)
            results.append(res)
    idx = np.argmin([r.fun for r in results])
    best_map = np.array(results)[idx]

    with sb.axes_style('white'):
        fig, axs = pl.subplots(1, 2, figsize=(10,3), sharex=True, sharey=True)
        axs[0].plot(t, f, 'k.')
        axs[0].plot(t, spz.model(get_theta(initial, 'sp'), *args[:-3]), 'b-', lw=5, label='initial')
        axs[0].plot(t, spz.model(get_theta(best_map.x, 'sp'), *args[:-3]), 'r-', lw=5, label='optimized')
        axs[0].legend()
        axs[1].plot(t, f-spz.model(get_theta(best_map.x, 'sp'), *args[:-3], ret_sys=True), 'k.')
        # axs[1].plot(t, spz.model(get_theta(initial, 'sp'), *args[:-3], ret_ma=True), 'b-', lw=5)
        axs[1].plot(t, spz.model(get_theta(best_map.x, 'sp'), *args[:-3], ret_ma=True), 'r-', lw=5)
        pl.setp(axs, xlim=[t.min(), t.max()], xticks=[], yticks=[])
        pl.setp(axs[0], title='raw')
        pl.setp(axs[1], title='corrected')
        fig.tight_layout()
        fp = os.path.join(out_dir, 'fit-map.png')
        fig.savefig(fp)
        pl.close()


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
            if (gr < gr_threshold).all():
                break

        fp = os.path.join(out_dir, 'gr.png')
        plot.gr_iter(gr_vals, fp)


        labels = 'ks,kk,tc,a,i,u1s,u2s,u1k,u2k,t0,sig,k0,k1'.split(',')
        labels += ['c{}'.format(i) for i in range(len(aux))]
        fp = os.path.join(out_dir, 'chains.png')
        plot.chain(sampler.chain, labels, fp)


        burn = nsteps - 1000 if nsteps > 2000 else 0
        thin = 1
        fc = sampler.chain[:,burn::thin,:].reshape(-1, ndim)
        fp = os.path.join(out_dir, 'corner.png')
        plot.corner(fc, labels, fp)


        maxprob = sampler.lnprobability.flatten().max()
        idx = np.argmax(sampler.lnprobability)
        assert sampler.lnprobability.flat[idx] == maxprob
        best = sampler.flatchain[idx]

        fp = os.path.join(out_dir, 'opt.txt')
        with open(fp, 'w') as o:
            o.write("MAP log prob: {}".format(-best_map.fun))
            o.write("\n\tparams: ")
            o.write(' '.join([str(i) for i in best_map.x]))
            o.write("\nMCMC log prob: {}".format(maxprob))
            o.write("\n\tparams: ")
            o.write(' '.join([str(i) for i in best]))

        best_sp = get_theta(best, 'sp')
        mod_full = spz.model(best_sp, *args[:-3])
        f_cor = f - spz.model(best_sp, *args[:-3], ret_sys=True)
        mod_ma = spz.model(best_sp, *args[:-3], ret_ma=True)
        resid = f - spz.model(best_sp, *args[:-3])
        fp = os.path.join(out_dir, 'fit-mcmc-best.png')
        plot.corrected_ts(t, f, f_cor, mod_full, mod_ma, resid, fp)


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


    best_sp = get_theta(best, 'sp')
    df_sp['f_cor'] = f - spz.model(best_sp, *args[:-3], ret_sys=True)

    tc = np.median(fc[:,2])
    spz_phase = list(t - tc)

    percs = [50, 16, 84]
    npercs = len(percs)

    flux_pr_k2, flux_pr_sp = [], []
    for theta in fc[np.random.permutation(fc.shape[0])[:1000]]:

        theta_sp = get_theta(theta, 'sp')
        theta_k2 = get_theta(theta, 'k2')

        flux_pr_sp.append(spz.model(theta_sp, t, f, s, p, aux, ret_ma=True))
        flux_pr_k2.append(k2.loglike(theta_k2, k2data[0], k2data[1], p, ret_ma=True))

    flux_pr_sp, flux_pr_k2 = map(np.array, [flux_pr_sp, flux_pr_k2])
    flux_pc_sp = np.percentile(flux_pr_sp, percs, axis=0)
    flux_pc_k2 = np.percentile(flux_pr_k2, percs, axis=0)

    fp = os.path.join(out_dir, 'fit-final.png')
    plot.k2_spz_together(df_sp, df_k2, spz_phase, flux_pc_sp, flux_pc_k2, percs, fc, fp)

    fp = os.path.join(out_dir, 'spz.csv')
    df_sp.to_csv(fp, index=False)