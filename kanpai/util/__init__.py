import yaml

import stats
import tables
import transit
import ts
import lc
import ld


def parse_setup(fp):
    setup = yaml.load(open(fp))
    tr = setup['transit']
    if 'b' not in tr.keys() or tr['b'] is None:
        tr['b'] = 0
    if 't14' not in tr.keys():
        try:
            tr['t14'] = transit.t14_circ(tr['p'],
                tr['a'], tr['k'], tr['b'])
        except KeyError as e:
            msg = "{} is missing! unable to compute transit duration"
            print(msg.format(e))
    if 'a' not in tr.keys() or tr['a'] is None:
        try:
            p = tr['p']
            t14 = tr['t14']
            k = tr['k']
            tr['a'] = transit.scaled_a(p, t14, k)
        except KeyError as e:
            msg = "{} is missing! unable to compute scaled semi-major axis"
            print(msg.format(e))
    if 'i' not in tr.keys() or tr['i'] is None:
        a, b = tr['a'], tr['b']
        tr['i'] = float(transit.inclination(a, b))
    setup['transit'] = tr
    return setup
