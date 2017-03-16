import numpy as np

from . import mod


def loglike_u(theta, t, f, p, ret_mod=False, sc=False):
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


def loglike_q(theta, t, f, p, ret_mod=False, sc=False):
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



def loglike_u_tc(theta, t, f, k, a, i, u1, u2, p, ret_mod=False, sc=False):
    tc,ls,k0 = theta
    if not sc:
        m = mod.model_u_tc(tc, t, k, a, i, u1, u2, p) + k0
    else:
        m = mod.model_u_tc_sc(tc, t, k, a, i, u1, u2, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = np.exp(-2*ls)
    return -0.5*(np.sum((resid)**2 * inv_sig2 + 2*ls))
