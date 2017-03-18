import numpy as np



def q_to_u(q1, q2):
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2*q2)
    return u1, u2


def u_to_q(u1, u2):
    q1 = (u1 + u2)**2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2



def claret(teff, uteff, logg, ulogg, feh, ufeh, band):

    mult = 1
    u = np.repeat(np.nan, 4)
    while np.isnan(u).any():
        u[:] = limbdark.get_ld(band, teff, mult * uteff, logg, mult * ulogg, feh, mult * ufeh)
        mult += 1

    u = u.tolist()
    # boost uncertainties by factor of 2
    u[1] *= 2
    u[3] *= 2

    print "{0} u1: {1:.4f} +/- {2:.4f}, u2: {3:.4f} +/- {4:.4f}".format(band, *u)

    df = limbdark.get_ld_df(band, teff, mult * uteff, logg, mult * ulogg, feh, mult * ufeh)
    print "Using {} models".format(df.shape[0])
    for key in "teff logg feh".split():
        print "{} range: {} - {}".format(key, df[key].min(), df[key].max())

    return u
