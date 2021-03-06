from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import sklearn.decomposition as dc
from astropy.stats import sigma_clip
from . import ts
from six.moves import range

def gelman_rubin(chains, verbose=False):
    assert chains.ndim == 3
    nn = chains.shape[1]
    mean_j = chains.mean(axis=1)
    var_j = chains.var(axis=1)
    B = nn * mean_j.var(axis=0)
    W = var_j.mean(axis=0)
    R2 = ( W*(nn-1)/nn + B/nn ) / W
    return np.sqrt(R2)


def geom_mean(x):
    x = np.abs(x)
    gm = np.sqrt(np.product(x)) if x.size > 1 else x
    return gm


def rms(x):
    return np.sqrt((x**2).sum()/x.size)


def bic(lnlike, ndata, nparam):
    return -2 * lnlike + nparam * np.log(ndata)


def chisq(resid, sig, ndata=None, nparams=None, reduced=False):
    if reduced:
        assert ndata is not None and nparams is not None
        dof = ndata - nparams
        return sum((resid / sig)**2) / (dof)
    else:
        return sum((resid / sig)**2)


def pca(X, n=2):

    pca = dc.PCA()
    res = pca.fit(X)
    ratio_exp = pca.explained_variance_ratio_
    for i in range(n):
        print("PCA BV{0} explained variance: {1:.4f}".format(i+1, ratio_exp[i]))

    return pca.components_[:n].T


def outliers(x, iterative=True, su=4, sl=4):

    if iterative:

        clip = sigma_clip(x, sigma_upper=su, sigma_lower=sl)
        idx = clip.mask

    else:

        mu, sig = np.median(x), np.std(x)
        idx = (x > mu + su * sig) | (x < mu - sl * sig)

    return idx


def beta(residuals, timestep, start_min=5, stop_min=20):

    """
    residuals : data - model
    timestep : time interval between datapoints in seconds
    """

    assert timestep < start_min * 60
    ndata = len(residuals)

    sigma1 = np.std(residuals)

    min_bs = int(start_min * 60 / timestep)
    max_bs = int(stop_min * 60 / timestep)

    betas = []
    for bs in range(min_bs, max_bs + 1):
        nbins = ndata / bs
        sigmaN_theory = sigma1 / np.sqrt(bs) * np.sqrt( nbins / (nbins - 1) )
        sigmaN_actual = np.std(ts.binned(residuals, bs))
        beta = sigmaN_actual / sigmaN_theory
        betas.append(beta)

    return np.median(betas)


def bin_std(residuals, timestep, width=20):

    """
    residuals : data - model
    timestep : time interval between datapoints in seconds
    width : bin width in minutes
    """

    ndata = len(residuals)
    bs = int(width * 60 / timestep)
    steps = range(0,ndata,bs) + [ndata]
    sigmas = [f[i:j].std() for i,j in zip(steps[:-1], steps[1:])]

    return np.median(sigmas)


def simple_ols(x, y, intercept=True):
    """
    Simple OLS with no y uncertainties.
    x : array-like, abscissa
    y : array-like, ordinate
    """
    if intercept:
        X = np.c_[np.ones_like(x), x]
    else:
        X = x
    return np.dot( np.dot( np.linalg.inv( np.dot(X.T, X) ), X.T), y )
