from .like import loglike
from .like import model
from fit import Fit
import util
import plot


def go(setup, method, bin_size, nsteps1, nsteps2, max_steps,
    gr_threshold, out_dir, save, nthreads, k2_kolded_fp, restart):

    fit = Fit(setup, out_dir, method, bin_size, k2_kolded_fp)
    fit.max_apo()
    fit.plot_max_apo()
    fit.run_mcmc(nthreads, nsteps1, nsteps2, max_steps, gr_threshold, save, restart)
    fit.plot_final()
