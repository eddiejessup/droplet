from __future__ import print_function
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse droplet distributions')
    parser.add_argument('R', type=float,
        help='Peak radius')
    parser.add_argument('fnames', nargs='+',
        help='Scatter event files')
    parser.add_argument('-t', type=float, default=0.0,
        help='Time to consider as steady state')
    args = parser.parse_args()

    for fname in args.fnames:
        t, r1, r2 = np.loadtxt(fname, delimiter=' ', unpack=True)

        r1 = r1[t > args.t]
        r2 = r2[t > args.t]

        r1p = r1 > args.R
        r2p = r2 > args.R

        n_pp = np.logical_and(r1p, r2p).sum()
        n_pp_err = np.sqrt(n_pp)

        n_pb = np.logical_and(r1p, np.logical_not(r2p)).sum()
        n_pb_err = np.sqrt(n_pb)

        n_pt = n_pb + n_pp
        n_pt_err = np.sqrt(n_pb_err ** 2 + n_pp_err ** 2)

        try:
            p_pb = float(n_pb) / n_pt
        except ZeroDivisionError:
            p_pb = np.nan
            p_pb_err = np.nan
        else:
            p_pb_err = p_pb * np.sqrt((n_pb_err / n_pb) ** 2 + (n_pt_err / n_pt) ** 2)

        datname = os.path.splitext(os.path.basename(fname))[0]
        print(datname, p_pb, p_pb_err)

        # for p, p_err in zip(ps, ps_err):
        #     print(datname, p, p_err)
        #     break
