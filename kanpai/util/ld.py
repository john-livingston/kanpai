from __future__ import absolute_import
import numpy as np

import limbdark


def q_to_u(q1, q2):
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2*q2)
    return u1, u2


def u_to_q(u1, u2):
    q1 = (u1 + u2)**2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2



def claret(teff, uteff, logg, ulogg, feh, ufeh, band):

    u = limbdark.claret(band, teff, uteff, logg, ulogg, feh, ufeh)

    # impose 10% minimum uncertainty on LD parameters
    u = list(u)
    if u[1]/u[0] < 0.1:
        u[1] = 0.1 * u[0]
    if u[3]/u[2] < 0.1:
        u[3] = 0.1 * u[2]

    return u
