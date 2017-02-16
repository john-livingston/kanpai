import numpy as np

from model import model1, model2


def loglike1(theta, t, f, s, p, ret_mod=False):
    k0 = theta[-1]
    m = model1(theta[:-1], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = s ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike2(theta, t, f, p, ret_mod=False):
    k0,sig = theta[-2:]
    m = model2(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = sig ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))
