import itertools
import numpy as np
import statsmodels.api as sm

from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter
import limbdark

from ..k2 import band


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

    print "{0} u1: {1:.4f} +/- {2:.4f}, u2: {3:.4f} +/- {4:.4f}".format(band, *u)

    df = limbdark.get_ld_df(band, teff, mult * uteff, logg, mult * ulogg)
    print "Using {} models".format(df.shape[0])
    for key in "teff logg feh".split():
        print "{} range: {} - {}".format(key, df[key].min(), df[key].max())

    return u


def get_ld_ldtk(teff, uteff, logg, ulogg, feh, ufeh, kind='quadratic'):

    filters = [TabulatedFilter('Kepler', band.lam, band.tra)]
    sc = LDPSetCreator(teff=(teff,uteff), logg=(logg,ulogg), z=(feh,ufeh), filters=filters)
    ps = sc.create_profiles()

    if kind == 'quadratic':
        cq,eq = ps.coeffs_qd(do_mc=True)
        # as in Crossfield et al. 2016, multiply the uncertainties by 5
        eq *= 5
        u = list(itertools.chain.from_iterable(zip(cq[0],eq[0])))
        print "{0} u1: {1:.4f} +/- {2:.4f}, u2: {3:.4f} +/- {4:.4f}".format('Kp', *u)
    elif kind == 'linear':
        cq,eq = ps.coeffs_ln(do_mc=True)
        # as in Crossfield et al. 2016, multiply the uncertainties by 5
        eq *= 5
        u = [cq[0][0], eq[0]]
        print "{0} u: {1:.4f} +/- {2:.4f}".format('Kp', *u)
    else:
        raise ValueError('kind must be one of: (linear, quadratic)')

    return [float(i) for i in u]
