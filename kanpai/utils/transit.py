import numpy as np
from astropy import constants as c
from astropy import units as u

from general import outliers


def impact(a, i):
    return np.abs(a * np.cos(i))


def inclination(a, b):
    return np.arccos(b / a)


def tdur_circ(p, a, k, i=np.pi/2.):
    b = impact(a, i)
    alpha = np.sqrt( (k + 1) ** 2 - b ** 2 )
    return (p / np.pi) * np.arcsin( alpha / np.sin(i) / a )


def scaled_a(p, t14, k, i=np.pi/2.):
    numer = np.sqrt( (k + 1) ** 2 )
    denom = np.sin(i) * np.sin(t14 * np.pi / p)
    return float(numer / denom)


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
