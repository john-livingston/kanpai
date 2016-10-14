import numpy as np
from everest import Everest


def fold(t, f, p, t0, width=0.8, clip=False, bl=False, t14=0.2):

    idx = np.isnan(t) | np.isnan(f)
    t, f = t[~idx], f[~idx]
    tns = get_tns(t, p, t0)
    tf, ff = np.empty(0), np.empty(0)

    for i,tn in enumerate(tns):

        idx = (t > tn - width/2.) & (t < tn + width/2.)
        ti = t[idx]-tn
        fi = f[idx]
        fi /= np.nanmedian(fi)

        if bl:

            idx = (ti < -t14/2.) | (ti > t14/2.)
            assert np.isfinite(ti[idx]).all() & np.isfinite(fi[idx]).all()
            assert idx.sum() > 0

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
        fc = sigma_clip(ff, sigma_lower=10, sigma_upper=2)
        tf, ff = tf[~fc.mask], ff[~fc.mask]

    return tf, ff



def get_everest_folded(epic, p, t0, t14):

    star = Everest(epic)
    t, f = star.time, star.flux
    idx = np.isnan(t) | np.isnan(f)
    t, f = t[~idx], f[~idx]
    tf, ff = fold(t, f, p, t0, t14=t14, width=0.6, bl=True)

    return tf, ff
