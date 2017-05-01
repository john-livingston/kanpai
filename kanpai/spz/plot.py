from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
import scipy.optimize as op
from scipy import stats


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
