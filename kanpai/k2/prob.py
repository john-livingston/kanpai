import numpy as np
from scipy import stats

from . import like
from .. import util


def logprob1(theta, t, f, p, up=None, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,u1,u2,s,k0'.split(',')
    elif ret_mod:
        return like.loglike1(theta, t, f, p, ret_mod=True)

    k,tc,a,b,u1,u2,s,k0 = theta

    if u1 < 0 or u1 > 2 or u2 < -1 or u2 > 1 or b < 0 or b > 1+k:
        return -np.inf

    lp = 0
    if up is not None:
        lp += np.log(stats.norm.pdf(u1, up[0], up[1]))
        lp += np.log(stats.norm.pdf(u2, up[2], up[3]))

    ll = like.loglike1(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return lp + ll


def logprob2(theta, t, f, p, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,t14,i,u,s,k0'.split(',')
    elif ret_mod:
        return like.loglike2(theta, t, f, p, ret_mod=True)

    k,tc,t14,i,u,s,k0 = theta

    if u < 0 or u > 1 or i < 0 or i > np.pi/2:
        return -np.inf

    ll = like.loglike2(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll


def logprob3(theta, t, f, p, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,u,s,k0'.split(',')
    elif ret_mod:
        return like.loglike3(theta, t, f, p, ret_mod=True)

    k,tc,a,b,u,s,k0 = theta

    if u < 0 or u > 1 or b < 0 or b > 1+k:
        return -np.inf

    ll = like.loglike3(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll


def logprob4(theta, t, f, p, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,q1,q2,s,k0'.split(',')
    elif ret_mod:
        return like.loglike4(theta, t, f, p, ret_mod=True)

    k,tc,a,b,q1,q2,s,k0 = theta

    if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1 or b < 0 or b > 1+k:
        return -np.inf

    ll = like.loglike4(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll
