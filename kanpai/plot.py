import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
from corner import corner as triangle


def errorbar(t, f, s, fp=None, **kwargs):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.errorbar(t, f, s, marker='o', linestyle='none', **kwargs)
        pl.setp(ax, xlim=[t.min(), t.max()])
        if fp:
            fig.savefig(fp)
            pl.close()


def pixels(pix, fp=None):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(5,5))
        pl.imshow(pix, interpolation='none')
        if fp:
            fig.savefig(fp)
            pl.close()


def centroids(t, x, y, fp=None):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.plot(t, x, label='x', color='b')
        ax.plot(t, y, label='y', color='r')
        ax.legend()
        pl.setp(ax, xlim=[t.min(), t.max()])
        if fp:
            fig.savefig(fp)
            pl.close()


def simple_ts(t, f, fp=None, **kwargs):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.plot(t, f, marker='o', lw=0, **kwargs)
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
        ax.plot(gr_vals, 'k-')
        pl.setp(ax, xlabel='iterations', ylabel='mean G-R')
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def chain(chain, labels, fp=None):
    with sb.axes_style('white'):
        nwalkers, nsteps, ndim = chain.shape
        fig, axs = pl.subplots(ndim, 1, figsize=(15,ndim/1.5), sharex=True)
        [axs.flat[i].plot(c, drawstyle='steps', color='k', alpha=4./nwalkers) for i,c in enumerate(chain.T)]
        [pl.setp(axs.flat[i], ylabel=labels[i]) for i,c in enumerate(chain.T)]
        if fp:
            fig.savefig(fp)
            pl.close()


def corner(fc, labels, fp=None):
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
            pl.savefig(fp)
            pl.close()


def k2_spz_together(df_sp, df_k2, spz_phase, flux_pc_sp, flux_pc_k2, percs, fc,
    fp=None, alpha=0.8, dpi=256):

    npercs = len(percs)

    with sb.axes_style('whitegrid'):

        fig, axs = pl.subplots(1, 3, figsize=(10,3), sharex=False, sharey=False)

        axs.flat[0].plot(df_k2.t, df_k2.f, 'k.', alpha=alpha)
        [axs.flat[0].fill_between(df_k2.t, *flux_pc_k2[i:i+2,:], alpha=0.4,
            facecolor='b', edgecolor='b') for i in range(1,npercs-1,2)]
        axs.flat[0].plot(df_k2.t, flux_pc_k2[0], 'b-', lw=1.5)

        axs.flat[1].plot(df_sp.t, df_sp.f_cor, 'k.', alpha=alpha)
        [axs.flat[1].fill_between(df_sp.t, *flux_pc_sp[i:i+2,:], alpha=0.4,
        facecolor='r', edgecolor='r') for i in range(1,npercs-1,2)]
        axs.flat[1].plot(df_sp.t, flux_pc_sp[0], 'r-', lw=1.5)

        edgs, bins = np.histogram(fc[:,:2], bins=30)
        axs.flat[2].hist(fc[:,1], bins=bins, histtype='stepfilled',
            color='b', alpha=alpha, lw=0, label='K2')
        axs.flat[2].hist(fc[:,0], bins=bins, histtype='stepfilled',
            color='r', alpha=alpha, lw=0, label='Spitzer')

        ylim = axs.flat[1].get_ylim()
        pl.setp(axs.flat[0], xlim=[spz_phase[0], spz_phase[-1]], ylim=ylim)
        pl.setp(axs.flat[1], xlim=[df_sp.t.min(), df_sp.t.max()], ylim=ylim)
        # pl.setp(axs, xticks=[], yticks=[])

        fig.tight_layout()
        if fp:
            fig.savefig(fp, dpi=dpi)
            pl.close()
