import yaml

from .. import util


def parse_setup(fp):
    setup = yaml.load(open(fp))
    transit = setup['transit']
    if 'b' not in transit.keys() or transit['b'] is None:
        transit['b'] = 0
    if 'i' not in transit.keys() or transit['i'] is None:
        transit['i'] = util.transit.inclination(a, transit['b'])
    if 't14' not in transit.keys():
        try:
            transit['t14'] = util.transit.tdur_circ(transit['p'],
                transit['a'], transit['k'], transit['b'])
        except KeyError as e:
            msg = "{} is missing! unable to compute transit duration"
            print(msg.format(e))
    if 'a' not in transit.keys() or transit['a'] is None:
        try:
            p = transit['p']
            t14 = transit['t14']
            k = transit['k']
            transit['a'] = util.transit.scaled_a(p, t14, k)
        except KeyError as e:
            msg = "{} is missing! unable to compute scaled semi-major axis"
            print(msg.format(e))
    setup['transit'] = transit
    return setup
