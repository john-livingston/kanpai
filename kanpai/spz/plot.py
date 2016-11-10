import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
from corner import corner as triangle


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


def corrected_ts(t, f, f_cor, mod_full, mod_ma, resid, fp=None):
    with sb.axes_style('white'):
        fig, axs = pl.subplots(1, 3, figsize=(10,3), sharex=True, sharey=False)
        axs.flat[0].plot(t, f, 'k.')
        axs.flat[0].plot(t, mod_full, '-', lw=2)
        axs.flat[1].plot(t, f_cor, 'k.')
        axs.flat[1].plot(t, mod_ma, '-', lw=5)
        axs.flat[2].plot(t, resid, 'k.')
        pl.setp(axs, xlim=[t.min(), t.max()], xticks=[], yticks=[])
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def gr_iter(gr_vals, fp=None):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(7,3))
        iterations = np.arange(len(gr_vals)).astype(int)+1
        ax.plot(iterations, gr_vals, 'k-', lw=5, alpha=0.5)
        pl.setp(ax, xlabel='iterations', ylabel='G-R')
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


def corner(fc, labels, fp=None, dpi=96):
    hist_kwargs = dict(lw=2, alpha=0.5)
    title_kwargs = dict(fontdict=dict(fontsize=12))
    with sb.axes_style('white'):
        triangle(fc,
            labels=labels,
            hist_kwargs=hist_kwargs,
            title_kwargs=title_kwargs,
            show_titles=True,
            quantiles=[0.16,0.5,0.84],
            title_fmt='.4f')
        if fp:
            pl.savefig(fp, dpi=dpi)
            pl.close()


def k2_spz_together(df_sp, df_k2, flux_pc_sp, flux_pc_k2, percs,
    k_s, k_k, fp=None, title='', alpha=0.8, dpi=256):

    npercs = len(percs)

    rc = {'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.major.size': 3,
          'ytick.major.size': 3}

    with sb.axes_style('ticks', rc):

        fig, axs = pl.subplots(1, 3, figsize=(10,3), sharex=False, sharey=False)

        axs.flat[0].plot(df_k2.t * 24, df_k2.f, 'k.', alpha=alpha)
        [axs.flat[0].fill_between(df_k2.ti * 24, *flux_pc_k2[i:i+2,:], alpha=0.4,
            facecolor='b', edgecolor='b') for i in range(1,npercs-1,2)]
        axs.flat[0].plot(df_k2.ti * 24, flux_pc_k2[0], 'b-', lw=1.5)

        axs.flat[1].plot(df_sp.phase * 24, df_sp.f_cor, 'k.', alpha=alpha)
        [axs.flat[1].fill_between(df_sp.phase * 24, *flux_pc_sp[i:i+2,:], alpha=0.4,
        facecolor='r', edgecolor='r') for i in range(1,npercs-1,2)]
        axs.flat[1].plot(df_sp.phase * 24, flux_pc_sp[0], 'r-', lw=1.5)

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
        # pl.setp(axs, xticks=[], yticks=[])
        axs.flat[0].yaxis.get_major_formatter().set_useOffset(False)
        axs.flat[1].yaxis.get_major_formatter().set_useOffset(False)
        axs.flat[2].yaxis.get_major_formatter().set_useOffset(False)
        xrot = 0
        yrot = 0
        pl.setp(axs.flat[0].xaxis.get_majorticklabels(), rotation=xrot)
        pl.setp(axs.flat[1].xaxis.get_majorticklabels(), rotation=xrot)
        # pl.setp(axs.flat[2].xaxis.get_majorticklabels(), rotation=xrot)
        pl.setp(axs.flat[2].xaxis.get_majorticklabels(), rotation=30)
        pl.setp(axs.flat[0].yaxis.get_majorticklabels(), rotation=yrot)
        pl.setp(axs.flat[1].yaxis.get_majorticklabels(), rotation=yrot)
        pl.setp(axs.flat[2].yaxis.get_majorticklabels(), rotation=yrot)
        pl.setp(axs.flat[1], title=title)

        fig.tight_layout()
        if fp:
            fig.savefig(fp, dpi=dpi)
            pl.close()
