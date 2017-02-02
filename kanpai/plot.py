import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
from corner import corner as triangle
import scipy.optimize as op
from scipy import stats


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
