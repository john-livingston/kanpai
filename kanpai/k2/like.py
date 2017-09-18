from __future__ import absolute_import
import numpy as np

from . import mod


def loglike_u(theta, t, f, p, sc=False, ret_mod=False):
    ls,k0 = theta[-2:]
    if not sc:
        m = mod.model_u(theta[:-2], t, p) + k0
    else:
        m = mod.model_u_sc(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = np.exp(-2*ls)
    return -0.5*(np.sum((resid)**2 * inv_sig2 + 2*ls))


def loglike_q(theta, t, f, p, sc=False, ret_mod=False):
    ls,k0 = theta[-2:]
    if not sc:
        m = mod.model_q(theta[:-2], t, p) + k0
    else:
        m = mod.model_q_sc(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = np.exp(-2*ls)
    return -0.5*(np.sum((resid)**2 * inv_sig2 + 2*ls))

def loglike_q_fitp(theta, t, f, ret_mod=False):
    ls,k0 = theta[-2:]
    m = mod.model_q_fitp(theta[:-2], t) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = np.exp(-2*ls)
    return -0.5*(np.sum((resid)**2 * inv_sig2 + 2*ls))
