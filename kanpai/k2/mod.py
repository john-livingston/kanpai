import numpy as np

from pytransit import MandelAgol

# from ..util import transit
from .. import util

MA = MandelAgol(supersampling=8, exptime=0.02)
MA_SC = MandelAgol()


def model_u(theta, t, p):
    k,tc,a,b,u1,u2 = theta
    i = util.transit.inclination(a, b)
    m = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model_u_sc(theta, t, p):
    k,tc,a,b,u1,u2 = theta
    i = util.transit.inclination(a, b)
    m = MA_SC.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model_q(theta, t, p):
    k,tc,a,b,q1,q2 = theta
    i = util.transit.inclination(a, b)
    u1, u2 = util.ld.q_to_u(q1, q2)
    m = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model_q_sc(theta, t, p):
    k,tc,a,b,q1,q2 = theta
    i = util.transit.inclination(a, b)
    u1, u2 = util.ld.q_to_u(q1, q2)
    m = MA_SC.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model_u_tc(theta, t, k, a, i, u1, u2, p):
    tc = theta
    m = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model_u_tc_sc(theta, t, k, a, i, u1, u2, p):
    tc = theta
    m = MA_SC.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m
