import numpy as np
from scipy import stats

from . import like
from .. import util


def logprob_u(theta, t, f, p, up=None, sc=False, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,u1,u2,ls,k0'.split(',')
    elif ret_mod:
        return like.loglike_u(theta, t, f, p, ret_mod=True, sc=sc)

    k,tc,a,b,u1,u2,ls,k0 = theta

    if u1 < 0 or u1 > 2 or u2 < -1 or u2 > 1 or b < 0 or b > 1 or k < 0 or k > 1:
        return -np.inf

    lp = 0
    if up is not None:
        lp += np.log(stats.norm.pdf(u1, loc=up[0], scale=up[1]))
        lp += np.log(stats.norm.pdf(u2, loc=up[2], scale=up[3]))

    ll = like.loglike_u(theta, t, f, p, sc=sc)

    if np.isnan(ll).any():
        return -np.inf
    return lp + ll


def logprob_q(theta, t, f, p, up=None, sc=False, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'k,tc,a,b,q1,q2,ls,k0'.split(',')
    elif ret_mod:
        return like.loglike_q(theta, t, f, p, ret_mod=True, sc=sc)

    k,tc,a,b,q1,q2,ls,k0 = theta

    if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1 or b < 0 or b > 1 or k < 0 or k > 1:
        return -np.inf

    u1, u2 = util.ld.q_to_u(q1, q2)

    lp = 0
    if up is not None:
        lp += np.log(stats.norm.pdf(u1, loc=up[0], scale=up[1]))
        lp += np.log(stats.norm.pdf(u2, loc=up[2], scale=up[3]))

    ll = like.loglike_q(theta, t, f, p, sc=sc)

    if np.isnan(ll).any():
        return -np.inf
    return lp + ll


def logprob_u_tc(theta, t, f, k, a, i, u1, u2, p, sc=False, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        return 'tc,ls,k0'.split(',')
    elif ret_mod:
        return like.loglike_u_tc(theta, t, f, k, a, i, u1, u2, p, ret_mod=True, sc=sc)

    tc,ls,k0 = theta

    if tc < t[0] or tc > t[-1]:
        return -np.inf

    ll = like.loglike_u_tc(theta, t, f, k, a, i, u1, u2, p, sc=sc)

    if np.isnan(ll).any():
        return -np.inf
    return ll
