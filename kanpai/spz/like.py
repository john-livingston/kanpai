from __future__ import absolute_import
import numpy as np

from .. import util
from . import mod


def loglike_u(theta, t, f, p, aux, ret_mod=False):
    k,tc,a,b,u1,u2,ls,k1 = theta[:8]
    m = mod.model_u(theta, t, f, p, aux)
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = np.exp(-2*ls)
    return -0.5*(np.sum((resid)**2 * inv_sig2 + 2*ls))

def loglike_q(theta, t, f, p, aux, ret_mod=False):
    k,tc,a,b,q1,q2,ls,k1 = theta[:8]
    m = mod.model_q(theta, t, f, p, aux)
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = np.exp(-2*ls)
    return -0.5*(np.sum((resid)**2 * inv_sig2 + 2*ls))
