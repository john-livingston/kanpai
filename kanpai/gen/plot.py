from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
import scipy.optimize as op
from scipy import stats



def corrected_ts(t, f, f_cor, mod_full, mod_ma, resid, fp=None):

    rc = {'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'xtick.minor.size': 2,
          'ytick.minor.size': 2}

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
