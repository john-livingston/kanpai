import numpy as np
from pytransit import MandelAgol


MA = MandelAgol(supersampling=8, exptime=0.02)


def loglike(theta, t, f, p, ret_ma=False):
    k,tc,a,i,u1,u2,sig = theta
    m = MA.evaluate(t, k, [u1, u2], tc, p, a, i)
    if ret_ma:
        return m
    resid = f - m
    inv_sigma2 = 1.0/(sig**2)
    return -0.5*(np.sum((resid)**2*inv_sigma2 - np.log(inv_sigma2)))
