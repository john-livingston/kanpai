import numpy as np
from pytransit import MandelAgol

from .. import util

MA = MandelAgol()


def model_u(theta, t, f, p, aux, ret_ma=False, ret_sys=False):
    k,tc,a,b,u1,u2,ls,k1 = theta[:8]
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


def model_q(theta, t, f, p, aux, ret_ma=False, ret_sys=False):
    k,tc,a,b,q1,q2,ls,k1 = theta[:8]
    auxcoeff = theta[8:]
    i = util.transit.inclination(a, b)
    u1, u2 = util.ld.q_to_u(q1, q2)
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
