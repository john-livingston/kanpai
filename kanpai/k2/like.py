import numpy as np

from mod import model1, model2, model3, model4


def loglike1(theta, t, f, p, ret_mod=False):
    s,k0 = theta[-2:]
    m = model1(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike2(theta, t, f, p, ret_mod=False):
    s,k0 = theta[-2:]
    m = model2(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike3(theta, t, f, p, ret_mod=False):
    s,k0 = theta[-2:]
    m = model3(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike4(theta, t, f, p, ret_mod=False):
    s,k0 = theta[-2:]
    m = model4(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))
