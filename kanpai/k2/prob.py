import numpy as np
from scipy import stats

from . import like
from .. import util


def logprob_u(theta, t, f, p, up=None, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,u1,u2,s,k0'.split(',')
    elif ret_mod:
        return like.loglike_u(theta, t, f, p, ret_mod=True)

    k,tc,a,b,u1,u2,s,k0 = theta

    if u1 < 0 or u1 > 2 or u2 < -1 or u2 > 1 or b < 0 or b > 1+k:
        return -np.inf

    lp = 0
    if up is not None:
        lp += np.log(stats.norm.pdf(u1, loc=up[0], scale=up[1]))
        lp += np.log(stats.norm.pdf(u2, loc=up[2], scale=up[3]))

    ll = like.loglike_u(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return lp + ll


def logprob_q(theta, t, f, p, up=None, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,q1,q2,s,k0'.split(',')
    elif ret_mod:
        return like.loglike_q(theta, t, f, p, ret_mod=True)

    k,tc,a,b,q1,q2,s,k0 = theta

    if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1 or b < 0 or b > 1+k:
        return -np.inf

    u1, u2 = util.ld.q_to_u(q1, q2)

    lp = 0
    if up is not None:
        lp += np.log(stats.norm.pdf(u1, loc=up[0], scale=up[1]))
        lp += np.log(stats.norm.pdf(u2, loc=up[2], scale=up[3]))

    ll = like.loglike_q(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return lp + ll


def logprob3(theta, t, f, p, up=None, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,t14,i,u,s,k0'.split(',')
    elif ret_mod:
        return like.loglike3(theta, t, f, p, ret_mod=True)

    k,tc,t14,i,u,s,k0 = theta

    if u < 0 or u > 1 or i < 0 or i > np.pi/2:
        return -np.inf

    ll = like.loglike3(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll


def logprob4(theta, t, f, p, up=None, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,u,s,k0'.split(',')
    elif ret_mod:
        return like.loglike4(theta, t, f, p, ret_mod=True)

    k,tc,a,b,u,s,k0 = theta

    if u < 0 or u > 1 or b < 0 or b > 1+k:
        return -np.inf

    ll = like.loglike4(theta, t, f, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll


def logprob_u_tc(theta, t, f, k, a, i, u1, u2, p, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'tc,s,k0'.split(',')
    elif ret_mod:
        return like.loglike_u_tc(theta, t, f, k, a, i, u1, u2, p, ret_mod=True)

    ll = like.loglike_u_tc(theta, t, f, k, a, i, u1, u2, p)

    if np.isnan(ll).any():
        return -np.inf
    return ll
