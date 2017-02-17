import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
from corner import corner as triangle
import scipy.optimize as op
from scipy import stats

import util


def gr_iter(gr_vals, fp=None):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(7,3))
        iterations = np.arange(len(gr_vals)).astype(int)+1
        ax.plot(iterations, gr_vals, 'k-', lw=5, alpha=0.5)
        pl.setp(ax, xlabel='Iterations', ylabel='G-R',
            xlim=[iterations[0],iterations[-1]])
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def chain(chain, labels, fp=None, dpi=96):
    with sb.axes_style('white'):
        nwalkers, nsteps, ndim = chain.shape
        fig, axs = pl.subplots(ndim, 1, figsize=(15,ndim/1.5), sharex=True)
        [axs.flat[i].plot(c, drawstyle='steps', color='k', alpha=4./nwalkers) for i,c in enumerate(chain.T)]
        [pl.setp(axs.flat[i], ylabel=labels[i]) for i,c in enumerate(chain.T)]
        if fp:
            fig.savefig(fp, dpi=dpi)
            pl.close()


def corner(fc, labels, fp=None, truths=None, quantiles=[0.16,0.5,0.84],
    plot_datapoints=True, dpi=96, tight=False):

    # hist_kwargs = dict(lw=2, alpha=0.5)
    hist_kwargs = dict(lw=1, alpha=1)
    title_kwargs = dict(fontdict=dict(fontsize=12))
    with sb.axes_style('white', pl.rcParamsDefault):
        # n = fc.shape[1]
        # fig, axs = pl.subplots(n, n, figsize=(n*2, n*2))
        triangle(fc,
            # fig=fig,
            labels=labels,
            truths=truths,
            truth_color='b',
            smooth=1,
            smooth1d=1,
            plot_datapoints=plot_datapoints,
            data_kwargs={'alpha':0.01},
            hist_kwargs=hist_kwargs,
            title_kwargs=title_kwargs,
            show_titles=True,
            quantiles=quantiles,
            title_fmt='.4f')
        # pl.setp(axs, xlabel=[], ylabel=[])
        if fp:
            if tight:
                pl.tight_layout()
            pl.savefig(fp, dpi=dpi)
            pl.close()

def simple_ts(t, f, tmodel=None, model=None, fp=None, title="", **kwargs):
    with sb.axes_style('whitegrid'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.plot(t, f, linestyle='none', marker='o',
            color='b', alpha=0.5, mew=1, mec='k', **kwargs)
        if tmodel is not None and model is not None:
            ax.plot(tmodel, model, 'r-', mew=1, mec='k', **kwargs)
        elif model is not None:
            ax.plot(t, model, 'r-', mew=1, mec='k', **kwargs)
        pl.setp(ax, xlabel='Time [BJD]',
            ylabel='Normalized Flux',
            title=title,
            xlim=(t.min(), t.max()))
        ax.yaxis.get_major_formatter().set_useOffset(False)

        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def samples(t, f, ps, fp=None, title="", **kwargs):
    with sb.axes_style('whitegrid'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.plot(t, f, linestyle='none', marker='o',
            color='b', alpha=0.5, mew=1, mec='k', **kwargs)
        for s in ps:
            ax.plot(t, s, 'r-', alpha=0.1, **kwargs)
        pl.setp(ax, xlabel='Time [BJD]',
            ylabel='Normalized Flux',
            title=title,
            xlim=(t.min(), t.max()))
        ax.yaxis.get_major_formatter().set_useOffset(False)

        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def cred_reg(t, f, ps, fp=None, title="", **kwargs):

    rc = {'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'xtick.minor.size': 2,
          'ytick.minor.size': 2}

    dfmt = 'k.'
    bfmt = 'k.'
    dms = 5
    bms = 10
    data_alpha = 0.6

    with sb.axes_style('ticks', rc):

        fig, ax = pl.subplots(1, 1, figsize=(5,3))

        axs.flat[0].plot(t, f, dfmt, ms=dms, alpha=data_alpha)
        if plot_binned:
            tkb, fkb = util.ts.binned_ts(t, f, 0.5, fun=np.median)
            axs.flat[0].plot(tkb, fkb, bfmt, ms=bms)
        [axs.flat[0].fill_between(df_k2.ti*24, *flux_pc_k2[i:i+2,:], alpha=0.5,
            facecolor='b', edgecolor='b') for i in range(1,npercs-1,2)]
        axs.flat[0].plot(df_k2.ti*24, flux_pc_k2[0], 'b-', lw=1)

        pl.setp(ax, xlabel='Hours from mid-transit', ylabel='Normalized flux')
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.minorticks_on()
        # xrot = 0
        # yrot = 0
        # pl.setp(ax.xaxis.get_majorticklabels(), rotation=xrot)
        # pl.setp(ax.yaxis.get_majorticklabels(), rotation=yrot)
        pl.setp(ax, title=title)

        fig.tight_layout()
        if fp:
            fig.savefig(fp, dpi=dpi)
            pl.close()


def multi_gauss_fit(samples, p0, fp=None, return_popt=False, verbose=True):

    def multi_gauss(x, *args):
        n = len(args)
        assert n % 3 == 0
        g = np.zeros(len(x))
        for i in range(0,n,3):
            a, m, s = args[i:i+3]
            g += a * stats.norm.pdf(x, m, s)
        return g

    hist, edges = np.histogram(samples, bins=100, normed=True)
    bin_width = np.diff(edges).mean()
    x, y = edges[:-1] + bin_width/2., hist
    try:
        popt, pcov = op.curve_fit(multi_gauss, x, y, p0=p0)
    except RuntimeError as e:
        print e
        with sb.axes_style('white'):
            fig,ax = pl.subplots(1,1, figsize=(7,3))
            ax.hist(samples, bins=30, normed=True,
                    histtype='stepfilled', color='gray', alpha=0.6)
        return

    ncomp = len(p0)/3
    names = 'amp mu sigma'.split() * ncomp
    comp = []
    for i in range(ncomp):
        comp += (np.zeros(3) + i).astype(int).tolist()
    if verbose:
        print
        for i,(p,u) in enumerate(zip(popt, np.sqrt(np.diag(pcov)))):
            print "{0}{1}: {2:.6f} +/- {3:.6f}".format(names[i], comp[i], p, u)

    a_,mu_,sig_ =[],[],[]
    for i in range(len(p0)/3):
        a_.append(popt[i*3])
        mu_.append(popt[i*3+1])
        sig_.append(popt[i*3+2])

    with sb.axes_style('white'):
        fig,ax = pl.subplots(1,1, figsize=(7,3))
        ax.hist(samples, bins=edges, normed=True,
                histtype='stepfilled', color=cp[1], alpha=0.6)
        for a,m,s in zip(a_,mu_,sig_):
            ax.plot(x, a * stats.norm.pdf(x, m, s), linestyle='-', color=cp[0])
        ax.plot(x, multi_gauss(x, *popt), linestyle='--', color=cp[4], lw=3)
        pl.setp(ax, xlim=[x.min(), x.max()], yticks=[])
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()

    if return_popt:
        return popt
