import os
import yaml
import numpy as np
import pandas as pd
import astropy.units as u

import forecaster

SOLRAD = (u.Rsun / u.Rearth).to(u.dimensionless_unscaled)

def geom_mean(x):
    x = np.abs(x)
    gm = np.sqrt(np.product(x)) if x.size > 1 else x
    return gm

def aggregate_results(list_of_output_dirs):

    starids,planets,rs,urs,tf,utf = [],[],[],[],[],[]

    ars,uars_plus,uars_minus = [],[],[]
    inc,uinc_plus,uinc_minus = [],[],[]
    k_k2,uk_k2_plus,uk_k2_minus = [],[],[]
    k_sp,uk_sp_plus,uk_sp_minus = [],[],[]
    tc_sp,utc_sp_plus,utc_sp_minus = [],[],[]


    for d in list_of_output_dirs:

        fp = os.path.join(d, 'input.yaml')
        infile = yaml.load(open(fp))
        fp = os.path.join(d, 'output.yaml')
        outfile = yaml.load(open(fp))

        starids.append(infile['config']['starid'])
        planets.append(infile['config']['planet'])
        rs.append(infile['stellar']['rstar'][0])
        urs.append(infile['stellar']['rstar'][1])
        tf.append(infile['stellar']['teff'][0])
        utf.append(infile['stellar']['teff'][1])

        try:
            method = outfile['method']
            beta = outfile['spz']['beta']
            bic = outfile['spz']['bic']
            rchisq = outfile['spz']['reduced_chisq']

            a,b,c = outfile['percentiles']['a']
            ars.append(b)
            uars_plus.append(c-b)
            uars_minus.append(b-a)

            a,b,c = outfile['percentiles']['i']
            inc.append(b * 180/np.pi)
            uinc_plus.append((c-b) * 180/np.pi)
            uinc_minus.append((b-a) * 180/np.pi)

            a,b,c = outfile['percentiles']['k_k']
            k_k2.append(b)
            uk_k2_plus.append(c-b)
            uk_k2_minus.append(b-a)

            a,b,c = outfile['percentiles']['k_s']
            k_sp.append(b)
            uk_sp_plus.append(c-b)
            uk_sp_minus.append(b-a)

            a,b,c = outfile['percentiles']['tc_s']
            tc_sp.append(b)
            utc_sp_plus.append(c-b)
            utc_sp_minus.append(b-a)

        except KeyError as e:
            print fp, e

    df = pd.DataFrame(
        dict(
            starid=starids,
            planet=planets,
            rstar=rs,
            urstar=urs,
            teff=tf,
            uteff=utf,
            ars=ars,
            uars_plus=uars_plus,
            uars_minus=uars_minus,
            inc=inc,
            uinc_plus=uinc_plus,
            uinc_minus=uinc_minus,
            k_k2=k_k2,
            uk_k2_plus=uk_k2_plus,
            uk_k2_minus=uk_k2_minus,
            k_sp=k_sp,
            uk_sp_plus=uk_sp_plus,
            uk_sp_minus=uk_sp_minus,
            tc_sp=tc_sp,
            utc_sp_plus=utc_sp_plus,
            utc_sp_minus=utc_sp_minus,
            )
        )

    return df


def add_mass(df):

    df['re_sp'] = df['k_sp'] * df['rstar'] * SOLRAD
    df['re_k2'] = df['k_k2'] * df['rstar'] * SOLRAD

    df['uk_k2'] = df[['k_k2_plus', 'k_k2_minus']].apply(geom_mean, axis=1, raw=True)
    df['uk_sp'] = df[['k_sp_plus', 'k_sp_minus']].apply(geom_mean, axis=1, raw=True)

    urstar_frac = df['urstar'] / df['rstar']
    uk_k2_frac = df['uk_k2'] / df['k_k2']
    uk_sp_frac = df['uk_sp'] / df['k_sp']

    df['ure_k2'] = df['re_k2'] * (urstar_frac + uk_k2_frac)
    df['ure_sp'] = df['re_sp'] * (urstar_frac + uk_sp_frac)

    mass = [mr.Rstat2M(re, ure, unit='Earth') for re,ure in zip(df['re_k2'], df['ure_k2'])]
    umass2 = [k[1:] if k is not None else np.nan for k in mass]
    umass1 = [geom_mean(i) if np.isfinite(i).all() else np.nan for i in umass2]
    mass1 = [k[0] if k is not None else np.nan for k in mass]

    df['me_k2'] = mass1
    df['ume_k2'] = umass1
    df['ume_k2_2'] = umass2


def make_table1(df):
    cols = 'starid k_k2 uk_k2_plus uk_k2_minus k_sp uk_sp_plus uk_sp_minus'.split()
    lines = []
    for idx in df.index:
        vals = df.loc[idx][cols].tolist()
        text = '{0} & ${1:.4f}^(+{2:.4f})_(-{3:.4f})$ & ${4:.4f}^(+{5:.4f})_(-{6:.4f})$\\\\'
        line = text.format(*vals).replace('(','{').replace(')','}')
        lines.append(line)
    return '\n'.join(lines)


def make_table2(df):
    cols = 'starid ars uars_plus uars_minus inc uinc_plus uinc_minus'
    cols += ' k_k2 uk_k2_plus uk_k2_minus k_sp uk_sp_plus uk_sp_minus'
    cols += ' tc_sp utc_sp_plus utc_sp_minus'
    cols = cols.split()
    lines = []
    for idx in df.index:
        vals = df.loc[idx][cols].tolist()
        text = '{0} & ${1:.4f}^(+{2:.4f})_(-{3:.4f})$ & ${4:.4f}^(+{5:.4f})_(-{6:.4f})$'
        text += ' & ${7:.4f}^(+{8:.4f})_(-{9:.4f})$ & ${10:.4f}^(+{11:.4f})_(-{12:.4f})$'
        text += ' & ${13:.4f}^(+{14:.4f})_(-{15:.4f})$\\\\'
        line = text.format(*vals).replace('(','{').replace(')','}')
        lines.append(line)
    return '\n'.join(lines)


def save_to_latex(df, fp):
    df.to_latex(open('test_table.tex', 'w'))
