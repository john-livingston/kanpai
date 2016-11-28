import os
import yaml
import pandas as pd

def aggregate_results(list_of_output_dirs):

    starids,planets,rs,urs,tf,utf = [],[],[],[],[],[]
    k_k2,k_k2_plus,k_k2_minus = [],[],[]
    k_sp,k_sp_plus,k_sp_minus = [],[],[]

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

            a,b,c = outfile['percentiles']['k_k']
            k_k2.append(b)
            k_k2_plus.append(c-b)
            k_k2_minus.append(b-a)

            a,b,c = outfile['percentiles']['k_s']
            k_sp.append(b)
            k_sp_plus.append(c-b)
            k_sp_minus.append(b-a)
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
            k_k2=k_k2,
            k_k2_plus=k_k2_plus,
            k_k2_minus=k_k2_minus,
            k_sp=k_sp,
            k_sp_plus=k_sp_plus,
            k_sp_minus=k_sp_minus,
            )
        )

    return df


def make_table(aggregated_results):
    raise NotImplementedError

def save_to_latex(table, fp):
    raise NotImplementedError
