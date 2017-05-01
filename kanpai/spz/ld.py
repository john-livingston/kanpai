from __future__ import absolute_import
import functools
import numpy as np

from .. import util


claret = functools.partial(util.ld.claret, band='S2')
