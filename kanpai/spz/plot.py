import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
from corner import corner as triangle
import scipy.optimize as op
from scipy import stats

import util

ncolors = 5
cp = [sb.desaturate(pl.cm.gnuplot((j+1)/float(ncolors+1)), 0.75) for j in range(ncolors)]


def errorbar(t, f, s, fp=None, **kwargs):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.errorbar(t, f, s, marker='o', color='b', linestyle='none', **kwargs)
        pl.setp(ax, xlim=[t.min(), t.max()], xlabel='Time [BJD]',
            ylabel='Normalized flux')
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def pixels(pix, fp=None):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(5,5))
        ax.imshow(pix, interpolation='none')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def centroids(t, x, y, fp=None):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.plot(t, x, label='x', color='b')
        ax.plot(t, y, label='y', color='r')
        ax.legend()
        pl.setp(ax, xlim=[t.min(), t.max()], xlabel='Time [BJD]',
            ylabel='Centroid')
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def simple_ts(t, f, fp=None, **kwargs):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.plot(t, f, 'bo', **kwargs)
        pl.setp(ax, xlim=[t.min(), t.max()], xlabel='Time [BJD]',
            ylabel='Normalized flux')
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


# def corrected_ts(t, f, f_cor, mod_full, mod_ma, resid, fp=None):
#     with sb.axes_style('white'):
#         fig, axs = pl.subplots(1, 3, figsize=(10,3), sharex=True, sharey=False)
#         axs.flat[0].plot(t, f, 'k.')
#         axs.flat[0].plot(t, mod_full, '-', lw=2)
#         axs.flat[1].plot(t, f_cor, 'k.')
#         axs.flat[1].plot(t, mod_ma, '-', lw=5)
#         axs.flat[2].plot(t, resid, 'k.')
#         pl.setp(axs, xlim=[t.min(), t.max()], xticks=[], yticks=[])
#         fig.tight_layout()
#         if fp:
#             fig.savefig(fp)
#             pl.close()

def corrected_ts(t, f, f_cor, mod_full, mod_ma, resid, fp=None):

    rc = {'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'xtick.minor.size': 2,
          'ytick.minor.size': 2}

    # t_offset = int(t)
    # t_offset = 2450000
    # t -= t_offset

    with sb.axes_style('white', rc):
        fig, axs = pl.subplots(3, 1, figsize=(6,6), sharex=True, sharey=False)
        axs.flat[0].plot(t, f, 'ko', ms=5, alpha=0.6)
        # axs.flat[0].plot(t, mod_full, 'r-', lw=1, label='Transit + Systematics')
        axs.flat[0].plot(t, mod_full, 'r-', lw=1.5, label='Model')
        axs.flat[0].legend()
        axs.flat[1].plot(t, f_cor, 'ko', ms=5, alpha=0.6)
        axs.flat[1].plot(t, mod_ma, 'r-', lw=3, label='Transit')
        axs.flat[1].legend()
        axs.flat[2].plot(t, resid, 'ko', ms=5, alpha=0.6)
        axs.flat[0].yaxis.get_major_formatter().set_useOffset(False)
        axs.flat[1].yaxis.get_major_formatter().set_useOffset(False)
        axs.flat[2].xaxis.get_major_formatter().set_useOffset(False)
        axs.flat[0].minorticks_on()
        axs.flat[1].minorticks_on()
        axs.flat[2].minorticks_on()
        pl.setp(axs.flat[2].xaxis.get_majorticklabels(), rotation=20)
        pl.setp(axs.flat[0], title='Raw data', ylabel='Normalized flux')
        pl.setp(axs.flat[1], title='Corrected', ylabel='Normalized flux')
        pl.setp(axs.flat[2], title='Precision: {0:.0f} ppm'.format(resid.std()*1e6), ylabel='Residuals')
        # pl.setp(axs.flat[2], xlim=[t.min(), t.max()], xlabel='T-{} [BJD]'.format(t_offset))
        pl.setp(axs.flat[2], xlim=[t.min(), t.max()], xlabel='Time [BJD]')
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


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


def k2_spz_together(df_sp, df_k2, flux_pc_sp, flux_pc_k2, npercs,
    k_s, k_k, fp=None, title='', alpha=0.8, dpi=256, plot_binned=False):

    rc = {'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.major.size': 3,
          'ytick.major.size': 3}

    t_k, f_k = df_k2.t * 24, df_k2.f
    t_s, f_s = df_sp.phase * 24, df_sp.f_cor

    dfmt = 'k.'
    bfmt = 'k.'
    dms = 5
    bms = 10
    data_alpha = 0.6

    with sb.axes_style('ticks', rc):

        fig, axs = pl.subplots(1, 3, figsize=(10,3), sharex=False, sharey=False)

        axs.flat[0].plot(t_k, f_k, dfmt, ms=dms, alpha=data_alpha)
        if plot_binned:
            tkb, fkb = util.binned_ts(t_k, f_k, 0.5, fun=np.median)
            axs.flat[0].plot(tkb, fkb, bfmt, ms=bms)
        [axs.flat[0].fill_between(df_k2.ti*24, *flux_pc_k2[i:i+2,:], alpha=0.5,
            facecolor='b', edgecolor='b') for i in range(1,npercs-1,2)]
        axs.flat[0].plot(df_k2.ti*24, flux_pc_k2[0], 'b-', lw=1)

        axs.flat[1].plot(t_s, f_s, dfmt, ms=dms, alpha=data_alpha)
        if plot_binned:
            tsb, fsb = util.binned_ts(t_s, f_s, 0.5, fun=np.median)
            axs.flat[1].plot(tsb, fsb, bfmt, ms=bms)
        [axs.flat[1].fill_between(t_s, *flux_pc_sp[i:i+2,:], alpha=0.5,
        facecolor='r', edgecolor='r') for i in range(1,npercs-1,2)]
        axs.flat[1].plot(t_s, flux_pc_sp[0], 'r-', lw=1)

        edgs, bins = np.histogram(np.append(k_k, k_s), bins=30)
        axs.flat[2].hist(k_k, bins=bins, histtype='stepfilled',
            color='b', alpha=alpha, lw=0, label='K2', normed=True)
        axs.flat[2].hist(k_s, bins=bins, histtype='stepfilled',
            color='r', alpha=alpha, lw=0, label='Spitzer', normed=True)
        axs.flat[2].legend(loc=2)

        ylim = axs.flat[1].get_ylim()
        pl.setp(axs.flat[:2], xlim=[df_sp.phase.min() * 24, df_sp.phase.max() * 24],
            ylim=ylim, xlabel='Hours from mid-transit', ylabel='Normalized flux')
        pl.setp(axs.flat[2], xlabel=r'$R_p/R_{\star}$', ylabel='Probability density')
        axs.flat[0].yaxis.get_major_formatter().set_useOffset(False)
        axs.flat[1].yaxis.get_major_formatter().set_useOffset(False)
        axs.flat[2].yaxis.get_major_formatter().set_useOffset(False)
        axs.flat[0].minorticks_on()
        axs.flat[1].minorticks_on()
        axs.flat[2].minorticks_on()
        xrot = 0
        yrot = 0
        pl.setp(axs.flat[0].xaxis.get_majorticklabels(), rotation=xrot)
        pl.setp(axs.flat[1].xaxis.get_majorticklabels(), rotation=xrot)
        pl.setp(axs.flat[2].xaxis.get_majorticklabels(), rotation=30)
        pl.setp(axs.flat[0].yaxis.get_majorticklabels(), rotation=yrot)
        pl.setp(axs.flat[1].yaxis.get_majorticklabels(), rotation=yrot)
        pl.setp(axs.flat[2].yaxis.get_majorticklabels(), rotation=yrot)
        pl.setp(axs.flat[1], title=title)

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
