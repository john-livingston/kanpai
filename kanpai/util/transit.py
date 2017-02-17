import numpy as np
from astropy import constants as c
from astropy import units as u


def impact(a, i):
    return np.abs(a * np.cos(i))


def inclination(a, b):
    return np.arccos(b / a)


def tdur_circ(p, a, k, b):
    i = inclination(a, b)
    alpha = np.sqrt( (k + 1)**2 - b**2 )
    return (p / np.pi) * np.arcsin( alpha / np.sin(i) / a )


def scaled_a(p, t14, k, i=np.pi/2, b=0):
    numer = np.sqrt( (k + 1)**2 - b**2 )
    denom = np.sin(i) * np.sin(t14 * np.pi / p)
    return float(numer / denom)


def rhostar(p, a):
    """
    Eq.4 of http://arxiv.org/pdf/1311.1170v3.pdf. Assumes circular orbit.
    """
    p = p * u.d
    gpcc = u.g / u.cm ** 3
    rho_mks = 3 * np.pi / c.G / p ** 2 * a ** 3
    return rho_mks.to(gpcc)


def logg(rho, r):
    r = (r * u.R_sun).cgs
    gpcc = u.g / u.cm ** 3
    rho *= gpcc
    g = 4 * np.pi / 3 * c.G.cgs * rho * r
    return np.log10(g.value)


def rho(logg, r):
    r = (r * u.R_sun).cgs
    g = 10 ** logg * u.cm / u.s ** 2
    rho = 3 * g / (r * c.G.cgs * 4 * np.pi)
    return rho


def sample_rhostar(a_samples, p):
    """
    Given samples of the scaled semi-major axis and the period,
    compute samples of rhostar
    """
    rho = []
    n = int(1e4) if len(a_samples) > 1e4 else len(a_samples)
    for a in a_samples[np.random.randint(len(a_samples), size=n)]:
        rho.append(rhostar(p, a).value)
    return np.array(rho)


def sample_logg(rho_samples, rstar, urstar):
    """
    Given samples of the stellar density and the stellar radius
    (and its uncertainty), compute samples of logg
    """
    rs = rstar + urstar * np.random.randn(len(rho_samples))
    idx = rs > 0
    return logg(rho_samples[idx], rs[idx])
