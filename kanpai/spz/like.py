import numpy as np
from pytransit import MandelAgol
# from scipy import stats

from .. import util

MA = MandelAgol()


def model(theta, t, f, p, aux, ret_ma=False, ret_sys=False):
    k,tc,a,b,u1,u2,s,k1 = theta[:8]
    auxcoeff = theta[8:]
    i = util.transit.inclination(a, b)
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


def loglike(theta, t, f, p, aux):
    k,tc,a,b,u1,u2,s,k1 = theta[:8]
    m = model(theta, t, f, p, aux)
    resid = f - m
    inv_sigma2 = 1.0/(s**2)
    return -0.5*(np.sum((resid)**2*inv_sigma2 - np.log(inv_sigma2)))


def logprob(theta, t, f, p, aux, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        pvn = 'a,b,k,tc,u1,u2,sig,k1'.split(',')
        if aux is not None:
            pvn += ['c{}'.format(i) for i in range(len(aux))]
        return pvn

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
