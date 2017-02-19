import numpy as np

from .. import util
from mod import model1, model2


def loglike1(theta, t, f, p, aux, ret_mod=False):
    k,tc,a,b,u1,u2,s,k1 = theta[:8]
    m = model1(theta, t, f, p, aux)
    if ret_mod:
        return m
    resid = f - m
    inv_sigma2 = 1.0/(s**2)
    return -0.5*(np.sum((resid)**2*inv_sigma2 - np.log(inv_sigma2)))


def loglike2(theta, t, f, p, aux, ret_mod=False):
    k,tc,a,b,q1,q2,s,k1 = theta[:8]
    m = model2(theta, t, f, p, aux)
    if ret_mod:
        return m
    resid = f - m
    inv_sigma2 = 1.0/(s**2)
    return -0.5*(np.sum((resid)**2*inv_sigma2 - np.log(inv_sigma2)))
