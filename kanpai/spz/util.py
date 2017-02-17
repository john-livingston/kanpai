import yaml

from .. import util


def parse_setup(fp):
    setup = yaml.load(open(fp))
    transit = setup['transit']
    if not transit['t14']:
        try:
            transit['t14'] = util.transit.tdur_circ(transit['p'],
                transit['a'], transit['k'], transit['b'])
        except KeyError as e:
            msg = "{} is missing! unable to compute transit duration"
            print(msg.format(e))
    if not transit['a']:
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
