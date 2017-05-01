from __future__ import absolute_import
from __future__ import print_function
import functools
import itertools
import numpy as np
import statsmodels.api as sm

from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter

from .. import util
from . import band
from six.moves import zip


claret = functools.partial(util.ld.claret, band='Kp')


def ldtk(teff, uteff, logg, ulogg, feh, ufeh, kind='quadratic'):

    filters = [TabulatedFilter('Kepler', band.lam, band.tra)]
    sc = LDPSetCreator(teff=(teff,uteff), logg=(logg,ulogg), z=(feh,ufeh), filters=filters)
    ps = sc.create_profiles()

    if kind == 'quadratic':
        cq,eq = ps.coeffs_qd(do_mc=True)
        # as in Crossfield et al. 2016, multiply the uncertainties by 5
        eq *= 5
        u = list(itertools.chain.from_iterable(list(zip(cq[0],eq[0]))))
        print("{0} u1: {1:.4f} +/- {2:.4f}, u2: {3:.4f} +/- {4:.4f}".format('Kp', *u))
    elif kind == 'linear':
        cq,eq = ps.coeffs_ln(do_mc=True)
        # as in Crossfield et al. 2016, multiply the uncertainties by 5
        eq *= 5
        u = [cq[0][0], eq[0]]
        print("{0} u: {1:.4f} +/- {2:.4f}".format('Kp', *u))
    else:
        raise ValueError('kind must be one of: (linear, quadratic)')

    return [float(i) for i in u]
