import numpy as np
from astropy.stats import sigma_clip


def binned(a, binsize, fun=np.mean):
    return np.array([fun(a[i:i+binsize], axis=0) \
        for i in range(0, a.shape[0], binsize)])


def binned_ts(t, x, w, fun=np.mean):
    """
    What one normally wants when binning time-series data.

    :param t    : the abscissa
    :param x    : the ordinate
    :param w    : the desired bin width (in same units as t)
    """

    nbins = int(round((t.max()-t.min())/w))
    bins = np.linspace(t.min(), t.max(), nbins)
    idx = np.digitize(t, bins)
    tb = [fun(t[idx==bn]) for bn in range(1,nbins)]
    xb = [fun(x[idx==bn]) for bn in range(1,nbins)]

    return np.array(tb), np.array(xb)


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
