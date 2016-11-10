import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
sb.set_color_codes('muted')
from corner import corner as triangle

import util


def simple_ts(t, f, tmodel=None, model=None, fp=None, title="", **kwargs):
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(10,3))
        ax.plot(t, f, 'bo', **kwargs)
        if tmodel is not None and model is not None:
            ax.plot(tmodel, model, 'r-', **kwargs)
        elif model is not None:
            ax.plot(t, model, 'r-', **kwargs)
        pl.setp(ax, xlabel='Time [BJD]',
            ylabel='Normalized Flux',
            title=title)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        fig.tight_layout()
        if fp:
            fig.savefig(fp)
            pl.close()
