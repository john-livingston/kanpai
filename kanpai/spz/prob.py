import numpy as np
from scipy import stats

from . import like
from .. import util


def logprob_u(theta, t, f, p, aux=None, up=None, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        pvn = 'k,tc,a,b,u1,u2,s,k1'.split(',')
        if aux is not None:
            pvn += ['c{}'.format(i) for i in range(len(aux))]
        return pvn
    elif ret_mod:
        return like.loglike_u(theta, t, f, p, aux, ret_mod=True)

    k,tc,a,b,u1,u2,s,k1 = theta[:8]

    if u1 < 0 or u1 > 2 or u2 < -1 or u2 > 1 or b < 0 or b > 1+k:
        return -np.inf

    lp = 0
    if up is not None:
        lp += np.log(stats.norm.pdf(u1, loc=up[0], scale=up[1]))
        lp += np.log(stats.norm.pdf(u2, loc=up[2], scale=up[3]))

    ll = like.loglike_u(theta, t, f, p, aux)

    if np.isnan(ll).any():
        return -np.inf
    return lp + ll


def logprob_q(theta, t, f, p, aux=None, up=None, ret_pvnames=False, ret_mod=False):

    if ret_pvnames:
        pvn = 'k,tc,a,b,q1,q2,s,k1'.split(',')
        if aux is not None:
            pvn += ['c{}'.format(i) for i in range(len(aux))]
        return pvn
    elif ret_mod:
        return like.loglike_q(theta, t, f, p, aux, ret_mod=True)

    k,tc,a,b,q1,q2,s,k1 = theta[:8]

    if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1 or b < 0 or b > 1+k:
        return -np.inf

    u1, u2 = util.ld.q_to_u(q1, q2)

    lp = 0
    if up is not None:
        lp += np.log(stats.norm.pdf(u1, loc=up[0], scale=up[1]))
        lp += np.log(stats.norm.pdf(u2, loc=up[2], scale=up[3]))

    ll = like.loglike_q(theta, t, f, p, aux)

    if np.isnan(ll).any():
        return -np.inf
    return lp + ll
