import numpy as np
import limbdark
import yaml


def scaled_a(p, t14, k, i=np.pi/2.):
    b = lambda *x: np.abs(x[0] * np.cos(x[1]))
    numer = np.sqrt( (k + 1) ** 2 )
    denom = np.sin(i) * np.sin(t14 * np.pi / p)
    return float(numer / denom)


def parse_setup(fp):
    setup = yaml.load(open(fp))
    transit = setup['transit']
    if not transit['i']:
        transit['i'] = np.pi/2
    if not transit['a']:
        try:
            p = transit['p']
            t14 = transit['t14']
            k = transit['k']
            i = transit['i']
            transit['a'] = scaled_a(p, t14, k, i)
        except KeyError as e:
            msg = "{} is missing! unable to compute scaled semi-major axis"
            print(msg.format(e))
    setup['transit'] = transit
    return setup


def get_ld(teff, uteff, logg, ulogg):
    mult = 1
    u_kep, u_spz = np.repeat(np.nan, 4), np.repeat(np.nan, 4)
    while np.isnan(u_kep).any() | np.isnan(u_spz).any():

        mult += 1

        u_kep[:] = limbdark.get_ld('Kp', teff, mult * uteff, logg, mult * ulogg)
        u_spz[:] = limbdark.get_ld('S2', teff, mult * uteff, logg, mult * ulogg)

    u_kep = u_kep.tolist()
    u_spz = u_spz.tolist()
    print "uncertainty multiplier needed: {}".format(mult)
    print "Kepler u1: {0:.4f}+/-{1:.4f}, u2: {2:.4f}+/-{3:.4f}".format(*u_kep)
    print "Spitzer u1: {0:.4f}+/-{1:.4f}, u2: {2:.4f}+/-{3:.4f}".format(*u_spz)

    df = limbdark.get_ld_df('Kp', teff, mult * uteff, logg, mult * ulogg)
    print "using {} models".format(df.shape[0])
    for key in "teff logg feh".split():
        print "{} range: {} - {}".format(key, df[key].min(), df[key].max())

    return u_kep, u_spz


def binned(a, binsize, fun=np.mean):
    return np.array([fun(a[i:i+binsize], axis=0) \
        for i in range(0, a.shape[0], binsize)])


def rms(x):
    return np.sqrt((x**2).sum()/x.size)


def beta(residuals, timestep, start_min=5, stop_min=20):

    """
    residuals : data - model
    timestep : time interval between datapoints in seconds
    """

    assert timestep < 300
    ndata = len(residuals)

    sigma1 = np.std(residuals)

    min_bs = int(start_min * 60 / timestep)
    max_bs = int(stop_min * 60 / timestep)

    betas = []
    for bs in range(min_bs, max_bs + 1):
        nbins = ndata / bs
        sigmaN_theory = sigma1 / np.sqrt(bs) * np.sqrt( nbins / (nbins - 1) )
        sigmaN_actual = np.std(binned(residuals, bs))
        beta = sigmaN_actual / sigmaN_theory
        betas.append(beta)

    return np.median(betas)


def piecewise_clip(f, lo, hi, nseg=16):

    width = f.size/nseg
    mask = np.array([]).astype(bool)
    for i in range(nseg):
        if i == nseg-1:
            fsub = f[i*width:]
        else:
            fsub = f[i*width:(i+1)*width]
        clipped = sigma_clip(fsub, sigma_lower=lo, sigma_upper=hi)
        mask = np.append(mask, clipped.mask)

    return mask
