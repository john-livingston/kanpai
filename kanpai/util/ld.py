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

    u = limbdark.claret_ld(band, teff, uteff, logg, ulogg, feh, ufeh)

    # boost uncertainties by factor of 2
    u[1] *= 2
    u[3] *= 2

    return u
