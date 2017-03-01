import numpy as np

from . import mod


def loglike_u(theta, t, f, p, ret_mod=False):
    s,k0 = theta[-2:]
    m = mod.model_u(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike_q(theta, t, f, p, ret_mod=False):
    s,k0 = theta[-2:]
    m = mod.model_q(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike3(theta, t, f, p, ret_mod=False):
    s,k0 = theta[-2:]
    m = mod.model3(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike4(theta, t, f, p, ret_mod=False):
    s,k0 = theta[-2:]
    m = mod.model4(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike_u_tc(theta, t, f, k, a, i, u1, u2, p, ret_mod=False):
    tc,s,k0 = theta
    m = mod.model_u_tc(tc, t, k, a, i, u1, u2, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))
