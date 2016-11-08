import numpy as np
from pytransit import MandelAgol

import util


MA = MandelAgol(supersampling=8, exptime=0.02)


def model1(theta, t, p):
    k,tc,a,i,u1,u2,_,_ = theta
    m = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model2(theta, t, p):
    k,tc,t14,i,u,_,_ = theta
    a = util.scaled_a(p, t14, k, i)
    m = MA.evaluate(t, k, (u, 0), tc, p, a, i)
    return m


def loglike1(theta, t, f, p, ret_mod=False):
    _,_,_,_,_,_,k0,sig = theta
    m = model1(theta, t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = sig ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))


def loglike2(theta, t, f, p, ret_mod=False):
    _,_,_,_,_,k0,sig = theta
    m = model2(theta, t, p) + k0
    if ret_mod:
        return m
    resid = f - m
    inv_sig2 = sig ** -2
    return -0.5*(np.sum((resid)**2 * inv_sig2 - np.log(inv_sig2)))
