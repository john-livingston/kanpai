import numpy as np

from pytransit import MandelAgol

# from ..util import transit
from .. import util

MA = MandelAgol(supersampling=8, exptime=0.02)


def model1(theta, t, p):
    k,tc,a,i,u1,u2 = theta
    m = MA.evaluate(t, k, (u1, u2), tc, p, a, i)
    return m


def model2(theta, t, p):
    k,tc,t14,i,u = theta
    a = util.transit.scaled_a(p, t14, k, i)
    m = MA.evaluate(t, k, (u, 0), tc, p, a, i)
    return m


def model3(theta, t, p):
    k,tc,a,b,u = theta
    i = util.transit.inclination(a, b)
    m = MA.evaluate(t, k, (u, 0), tc, p, a, i)
    return m
