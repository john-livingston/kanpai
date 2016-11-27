import itertools
import numpy as np
import statsmodels.api as sm
from astropy.stats import sigma_clip
from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter
import limbdark

from ..k2 import band


def rms(x):

    return np.sqrt((x**2).sum()/x.size)


def scaled_a(p, t14, k, i=np.pi/2.):

    numer = np.sqrt( (k + 1) ** 2 )
    denom = np.sin(i) * np.sin(t14 * np.pi / p)

    return float(numer / denom)


def outliers(x, iterative=True, su=4, sl=4):

    if iterative:

        clip = sigma_clip(x, sigma_upper=su, sigma_lower=sl)
        idx = clip.mask

    else:

        mu, sig = np.median(x), np.std(x)
        idx = (x > mu + su * sig) | (x < mu - sl * sig)

    return idx


def get_tns(t, p, t0):

    idx = t != 0
    t = t[idx]

    while t0-p > t.min():
        t0 -= p
    if t0 < t.min():
        t0 += p

    tns = [t0+p*i for i in range(int((t.max()-t0)/p+1))]

    while tns[-1] > t.max():
        tns.pop()

    while tns[0] < t.min():
        tns = tns[1:]

    return tns


def fold(t, f, p, t0, t14=0.2, width=0.8, clip=False, bl=False, skip=None):

    idx = np.isnan(t) | np.isnan(f)
    t, f = t[~idx], f[~idx]
    tns = get_tns(t, p, t0)

    if skip:
        assert max(skip) < len(tns)
        for i in reversed(sorted(skip)):
            tns.pop(i)

    tf, ff = np.empty(0), np.empty(0)

    for i,tn in enumerate(tns):

        idx = (t > tn - width/2.) & (t < tn + width/2.)
        ti = t[idx]-tn
        fi = f[idx]
        fi /= np.nanmedian(fi)

        if bl:

            idx = (ti < -t14/2.) | (ti > t14/2.)
            if idx.sum() == 0:
                continue

            try:

                res = sm.RLM(fi[idx], sm.add_constant(ti[idx])).fit()

                if np.abs(res.params[1]) > 1e-2:
                    print "Bad data possibly causing poor fit"
                    print "Transit {} baseline params: {}".format(i, res.params)
                    continue

                model = res.params[0] + res.params[1] * ti
                fi = fi - model + 1

            except:

                print "Error computing baseline for transit {}".format(i)
                print "Num. datapoints: {}".format(idx.sum())
                print ti

        tf = np.append(tf, ti)
        ff = np.append(ff, fi / np.nanmedian(fi))

    idx = np.argsort(tf)
    tf = tf[idx]
    ff = ff[idx]

    if clip:
        idx = outliers(ff, su=clip[0], sl=clip[1])
        print "Clipped {} outliers".format(idx.sum())
        tf, ff = tf[~idx], ff[~idx]

    idx = (tf < -t14/2.) | (t14/2. < tf)
    print "OOT std dev: {}".format(ff[idx].std())

    return tf, ff


def get_ld_claret(teff, uteff, logg, ulogg, band='Kp'):

    mult = 1
    u = np.repeat(np.nan, 4)
    while np.isnan(u).any():
        mult += 1
        u[:] = limbdark.get_ld(band, teff, mult * uteff, logg, mult * ulogg)

    u = u.tolist()
    # boost uncertainties by factor of 2
    u[1] *= 2
    u[3] *= 2

    print "{0} u1: {1:.4f}+/-{2:.4f}, u2: {3:.4f}+/-{4:.4f}".format(band, *u)

    df = limbdark.get_ld_df(band, teff, mult * uteff, logg, mult * ulogg)
    print "Using {} models".format(df.shape[0])
    for key in "teff logg feh".split():
        print "{} range: {} - {}".format(key, df[key].min(), df[key].max())

    return u


def get_ld_ldtk(teff, uteff, logg, ulogg, feh, ufeh):

    filters = [TabulatedFilter('Kepler', band.lam, band.tra)]
    sc = LDPSetCreator(teff=(teff,uteff), logg=(logg,ulogg), z=(feh,ufeh), filters=filters)
    ps = sc.create_profiles()
    cq,eq = ps.coeffs_qd(do_mc=True)
    # as in Crossfield et al. 2016, multiply the uncertainties by 5
    eq *= 5
    u = list(itertools.chain.from_iterable(zip(cq[0],eq[0])))

    print "{0} u1: {1:.4f}+/-{2:.4f}, u2: {3:.4f}+/-{4:.4f}".format('Kp', *u)

    return u
