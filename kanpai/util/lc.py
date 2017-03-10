import numpy as np
import statsmodels.api as sm

from stats import outliers


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


def fold(t, f, p, t0, t14=0.2, width=0.8, clip=False, bl=False, skip=None, ret_seg=False, max_slope=1e-1):

    idx = np.isnan(t) | np.isnan(f)
    t, f = t[~idx], f[~idx]
    tns = get_tns(t, p, t0)

    orb = range(len(tns))

    if skip is not None:
        orb = [i for i in range(len(tns)) if i not in skip]
        assert max(skip) < len(tns)
        for i in reversed(sorted(skip)):
            tns.pop(i)

    assert len(orb) is len(tns)

    tf, ff = np.empty(0), np.empty(0)
    ts, fs = [], []

    for o,tn in zip(orb,tns):

        idx = (t > tn - width/2.) & (t < tn + width/2.)
        ti = t[idx]-tn
        fi = f[idx]
        fi /= np.nanmedian(fi)

        if bl:

            idx = (ti < -t14/2.) | (ti > t14/2.)
            if idx.sum() == 0:
                orb.pop(orb.index(o))
                print "No valid data for baseline fit"
                continue

            try:

                res = sm.RLM(fi[idx], sm.add_constant(ti[idx])).fit()

                if np.abs(res.params[1]) > max_slope:
                    print "Bad data possibly causing poor fit"
                    print "Transit {} baseline params: {}".format(i, res.params)
                    orb.pop(orb.index(o))
                    continue

                model = res.params[0] + res.params[1] * ti
                fi = fi - model + 1

            except:

                print "Error computing baseline for transit {}".format(i)
                print "Num. datapoints: {}".format(idx.sum())
                print ti
                orb.pop(orb.index(o))
                continue

        idx = (t > tn - width/2.) & (t < tn + width/2.)
        ts.append(t[idx].tolist())
        fs.append(fi.tolist())

        tf = np.append(tf, ti)
        ff = np.append(ff, fi / np.nanmedian(fi))

    if ret_seg:
        return orb, ts, fs

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
