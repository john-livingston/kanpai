import numpy as np
import statsmodels.api as sm
from astropy.stats import sigma_clip


def rms(x):
    return np.sqrt((x**2).sum()/x.size)


def scaled_a(p, t14, k, i=np.pi/2.):
    numer = np.sqrt( (k + 1) ** 2 )
    denom = np.sin(i) * np.sin(t14 * np.pi / p)
    return float(numer / denom)


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


def fold(t, f, p, t0, width=0.8, clip=False, bl=False, t14=0.2, skip=None):

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
                    print "bad data probably causing poor fit"
                    print "transit {} baseline params: {}".format(i, res.params)
                    continue

                model = res.params[0] + res.params[1] * ti
                fi = fi - model + 1

            except:

                print "error computing baseline for transit {}".format(i)
                print "num. points: {}".format(idx.sum())
                print ti

        tf = np.append(tf, ti)
        ff = np.append(ff, fi / np.nanmedian(fi))

    idx = np.argsort(tf)
    tf = tf[idx]
    ff = ff[idx]

    if clip:
        fc = sigma_clip(ff, sigma_upper=clip[0], sigma_lower=clip[1])
        print "clipped {} outliers".format(fc.mask.sum())
        tf, ff = tf[~fc.mask], ff[~fc.mask]

    idx = (tf < -t14/2.) | (t14/2. < tf)
    print "OOT std dev: {}".format(ff[idx].std())

    return tf, ff
