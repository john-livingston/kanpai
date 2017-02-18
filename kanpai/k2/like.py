import numpy as np

from model import model1, model2, model3, model4


def loglike1(theta, t, f, p, ret_mod=False):
    k0,sig = theta[-2:]
    m = model1(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = sig ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike2(theta, t, f, p, ret_mod=False):
    k0,sig = theta[-2:]
    m = model2(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = sig ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike3(theta, t, f, p, ret_mod=False):
    k0,sig = theta[-2:]
    m = model3(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = sig ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike4(theta, t, f, p, ret_mod=False):
    k0,sig = theta[-2:]
    m = model4(theta[:-2], t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = sig ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def logprob1(theta, t, f, p, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,u1,u2,k0,sig'.split(',')
    elif ret_mod:
        return loglike1(theta, t, f, p, ret_mod=True)

    k,tc,a,b,u1,u2,k0,sig = theta

    if u1 < 0 or u1 > 1 or u2 < 0 or u2 > 1 or b < 0 or b > 1+k:
        return -np.inf

    ll = loglike1(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll


def logprob2(theta, t, f, p, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,t14,i,u,k0,sig'.split(',')
    elif ret_mod:
        return loglike2(theta, t, f, p, ret_mod=True)

    k,tc,t14,i,u,k0,sig = theta

    if u < 0 or u > 1 or i < 0 or i > np.pi/2:
        return -np.inf

    ll = loglike2(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll


def logprob3(theta, t, f, p, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,u,k0,sig'.split(',')
    elif ret_mod:
        return loglike3(theta, t, f, p, ret_mod=True)

    k,tc,a,b,u,k0,sig = theta

    if u < 0 or u > 1 or b < 0 or b > 1+k:
        return -np.inf

    ll = loglike3(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll


def logprob4(theta, t, f, p, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,q1,q2,k0,sig'.split(',')
    elif ret_mod:
        return loglike4(theta, t, f, p, ret_mod=True)

    k,tc,a,b,q1,q2,k0,sig = theta

    if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1 or b < 0 or b > 1+k:
        return -np.inf

    ll = loglike4(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll
