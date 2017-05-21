from __future__ import absolute_import
import functools
import numpy as np

from .. import util

# band must be one of:
# B C H I J K Kp R S1 S2 S3 S4 U V b g* i* r* u u* v y z*
claret_J = functools.partial(util.ld.claret, band='J')
