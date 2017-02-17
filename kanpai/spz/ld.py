import numpy as np

import limbdark
from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter

from ..k2 import band as k2_band



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

    # FIXME move K2 LD completely out of this package
    filters = [TabulatedFilter('Kepler', k2_band.lam, k2_band.tra),
        # BoxcarFilter('I1', 3179, 3955),
        BoxcarFilter('I2', 3955, 5015)]
    sc = LDPSetCreator(teff=(teff,uteff), logg=(logg,ulogg), z=(feh,ufeh), filters=filters)
    ps = sc.create_profiles()
    cq,eq = ps.coeffs_qd(do_mc=True)

    return cq, eq
