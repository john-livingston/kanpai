import numpy as np

from pytransit import MandelAgol

# from ..util import transit
from .. import util

MA = MandelAgol(supersampling=8, exptime=0.02)


def model_u(theta, t, p):
    k,tc,a,b,u1,u2 = theta
    i = util.transit.inclination(a, b)
    m = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model_q(theta, t, p):
    k,tc,a,b,q1,q2 = theta
    i = util.transit.inclination(a, b)
    u1, u2 = util.ld.q_to_u(q1, q2)
    m = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model3(theta, t, p):
    k,tc,t14,i,u = theta
    a = util.transit.scaled_a(p, t14, k, i)
    m = MA.evaluate(t, k, (u, 0), tc, p, a, i)
    return m


def model4(theta, t, p):
    k,tc,a,b,u = theta
    i = util.transit.inclination(a, b)
    m = MA.evaluate(t, k, (u, 0), tc, p, a, i)
    return m


def model_u_tc(theta, t, k, a, i, u1, u2, p):
    tc = theta
    m = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m
