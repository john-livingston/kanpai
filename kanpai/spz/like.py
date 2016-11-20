import numpy as np
from pytransit import MandelAgol


MA = MandelAgol()


def model(theta, t, f, s, p, aux, ret_ma=False, ret_sys=False):
    k,tc,a,i,u1,u2,k1 = theta[:7]
    auxcoeff = theta[7:]
    ma = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    bl = k1 * (t-t[0])
    if aux is None:
        sys = 0
    elif aux.shape[0] == aux.size:
        sys = auxcoeff * aux
    else:
        sys = (auxcoeff * aux.T).sum(axis=1)
    if not ret_ma and not ret_sys:
        return ma + bl + sys
    if ret_sys:
        return bl + sys
    if ret_ma:
        return ma


def loglike(theta, t, f, s, p, aux):
    m = model(theta, t, f, s, p, aux)
    resid = f - m
    inv_sigma2 = 1.0/(s**2)
    return -0.5*(np.sum((resid)**2*inv_sigma2 - np.log(inv_sigma2)))
