import yaml
import numpy as np

import limbdark
from astropy import constants as c
from astropy import units as u
import sklearn.decomposition as dc
from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter

from ..k2 import band as k2_band



def impact(a, i):
    return np.abs(a * np.cos(i))


def tdur_circ(p, a, k, i=np.pi/2.):
    b = impact(a, i)
    alpha = np.sqrt( (k + 1) ** 2 - b ** 2 )
    return (p / np.pi) * np.arcsin( alpha / np.sin(i) / a )


def scaled_a(p, t14, k, i=np.pi/2.):
    numer = np.sqrt( (k + 1) ** 2 )
    denom = np.sin(i) * np.sin(t14 * np.pi / p)
    return float(numer / denom)


def parse_setup(fp):
    setup = yaml.load(open(fp))
    transit = setup['transit']
    if not transit['i']:
        transit['i'] = np.pi/2
    if not transit['t14']:
        try:
            transit['t14'] = tdur_circ(transit['p'],
                transit['a'], transit['k'], transit['i'])
        except KeyError as e:
            msg = "{} is missing! unable to compute transit duration"
            print(msg.format(e))
    if not transit['a']:
        try:
            p = transit['p']
            t14 = transit['t14']
            k = transit['k']
            i = transit['i']
            transit['a'] = scaled_a(p, t14, k, i)
        except KeyError as e:
            msg = "{} is missing! unable to compute scaled semi-major axis"
            print(msg.format(e))
    setup['transit'] = transit
    return setup


def get_ld_claret(teff, uteff, logg, ulogg, band='S2'):

    mult = 1
    u = np.repeat(np.nan, 4)
    while np.isnan(u).any():
        mult += 1
        u[:] = limbdark.get_ld(band, teff, mult * uteff, logg, mult * ulogg)

    u = u.tolist()
    # boost uncertainties by factor of 2
    u[1] *= 2
    u[3] *= 2

    print "{0} u1: {1:.4f} +/- {2:.4f}, u2: {3:.4f} +/- {4:.4f}".format(band, *u)

    df = limbdark.get_ld_df(band, teff, mult * uteff, logg, mult * ulogg)
    print "Using {} models".format(df.shape[0])
    for key in "teff logg feh".split():
        print "{} range: {} - {}".format(key, df[key].min(), df[key].max())

    return u


def get_ld_ldtk(teff, uteff, logg, ulogg, feh, ufeh):

    filters = [TabulatedFilter('Kepler', k2_band.lam, k2_band.tra),
        # BoxcarFilter('I1', 3179, 3955),
        BoxcarFilter('I2', 3955, 5015)]
    sc = LDPSetCreator(teff=(teff,uteff), logg=(logg,ulogg), z=(feh,ufeh), filters=filters)
    ps = sc.create_profiles()
    cq,eq = ps.coeffs_qd(do_mc=True)

    return cq, eq


def binned(a, binsize, fun=np.mean):
    return np.array([fun(a[i:i+binsize], axis=0) \
        for i in range(0, a.shape[0], binsize)])


def binned_ts(t, x, w, fun=np.mean):
    """
    What one normally wants when binning time-series data.

    :param t    : the abscissa
    :param x    : the ordinate
    :param w    : the desired bin width (in same units as t)
    """

    nbins = int(round((t.max()-t.min())/w))
    bins = np.linspace(t.min(), t.max(), nbins)
    idx = np.digitize(t, bins)
    tb = [fun(t[idx==bn]) for bn in range(1,nbins)]
    xb = [fun(x[idx==bn]) for bn in range(1,nbins)]

    return np.array(tb), np.array(xb)

def rms(x):
    return np.sqrt((x**2).sum()/x.size)


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
        sigmaN_actual = np.std(binned(residuals, bs))
        beta = sigmaN_actual / sigmaN_theory
        betas.append(beta)

    return np.median(betas)


def piecewise_clip(f, lo, hi, nseg=16):

    width = f.size/nseg
    mask = np.array([]).astype(bool)
    for i in range(nseg):
        if i == nseg-1:
            fsub = f[i*width:]
        else:
            fsub = f[i*width:(i+1)*width]
        clipped = sigma_clip(fsub, sigma_lower=lo, sigma_upper=hi)
        mask = np.append(mask, clipped.mask)

    return mask


def gelman_rubin(chains, verbose=False):
    assert chains.ndim == 3
    nn = chains.shape[1]
    mean_j = chains.mean(axis=1)
    var_j = chains.var(axis=1)
    B = nn * mean_j.var(axis=0)
    W = var_j.mean(axis=0)
    R2 = ( W*(nn-1)/nn + B/nn ) / W
    return np.sqrt(R2)


def pca(X, n=2):

    pca = dc.PCA()
    res = pca.fit(X)
    ratio_exp = pca.explained_variance_ratio_
    for i in range(n):
        print "PCA BV{0} explained variance: {1:.4f}".format(i+1, ratio_exp[i])

    return pca.components_[:n].T


def chisq(resid, sig, ndata=None, nparams=None, reduced=False):
    if reduced:
        assert ndata is not None and nparams is not None
        dof = ndata - nparams
        return sum((resid / sig)**2) / (dof)
    else:
        return sum((resid / sig)**2)


def bic(lnlike, ndata, nparam):
    return -2 * lnlike + nparam * np.log(ndata)


def rhostar(p, a):
    """
    Eq.4 of http://arxiv.org/pdf/1311.1170v3.pdf. Assumes circular orbit.
    """
    p = p * u.d
    gpcc = u.g / u.cm ** 3
    rho_mks = 3 * np.pi / c.G / p ** 2 * a ** 3
    return rho_mks.to(gpcc)


def logg(rho, r):
    r = (r * u.R_sun).cgs
    gpcc = u.g / u.cm ** 3
    rho *= gpcc
    g = 4 * np.pi / 3 * c.G.cgs * rho * r
    return np.log10(g.value)


def rho(logg, r):
    r = (r * u.R_sun).cgs
    g = 10 ** logg * u.cm / u.s ** 2
    rho = 3 * g / (r * c.G.cgs * 4 * np.pi)
    return rho


def sample_rhostar(a_samples, p):
    """
    Given samples of the scaled semi-major axis and the period,
    compute samples of rhostar
    """
    rho = []
    n = int(1e4) if len(a_samples) > 1e4 else len(a_samples)
    for a in a_samples[np.random.randint(len(a_samples), size=n)]:
        rho.append(rhostar(p, a).value)
    return np.array(rho)


def sample_logg(rho_samples, rstar, urstar):
    """
    Given samples of the stellar density and the stellar radius
    (and its uncertainty), compute samples of logg
    """
    rs = rstar + urstar * np.random.randn(len(rho_samples))
    idx = rs > 0
    return logg(rho_samples[idx], rs[idx])
