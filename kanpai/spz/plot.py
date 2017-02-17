import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
import scipy.optimize as op
from scipy import stats

import util
from ..plot import corner


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

    # t_offset = int(t[0])
    # t_offset = 2450000
    # t -= t_offset

    with sb.axes_style('white', rc):
        fig, axs = pl.subplots(3, 1, figsize=(6,6), sharex=True, sharey=False)
        axs.flat[0].plot(t, f, 'ko', ms=5, alpha=0.6)
        # axs.flat[0].plot(t, mod_full, 'r-', lw=1, label='Transit + Systematics')
        axs.flat[0].plot(t, mod_full, 'r-', lw=1.5, label='Model')
        # axs.flat[0].legend()
        axs.flat[1].plot(t, f_cor, 'ko', ms=5, alpha=0.6)
        axs.flat[1].plot(t, mod_ma, 'r-', lw=3, label='Transit')
        # axs.flat[1].legend()
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
        # pl.setp(axs.flat[2], title='Precision: {0:.0f} ppm'.format(resid.std()*1e6), ylabel='Residuals')
        pl.setp(axs.flat[2], title='Residuals')
        # pl.setp(axs.flat[2], xlim=[t.min(), t.max()], xlabel='T-{} [BJD]'.format(t_offset))
        pl.setp(axs.flat[2], xlim=[t.min(), t.max()], xlabel='Time [BJD]')
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()


def k2_spz_together(df_sp, df_k2, flux_pc_sp, flux_pc_k2, npercs, k_s, k_k,
    fp=None, title='', alpha=0.8, dpi=256, plot_binned=False, plot_depth=False):

    rc = {'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'xtick.minor.size': 2,
          'ytick.minor.size': 2}

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
