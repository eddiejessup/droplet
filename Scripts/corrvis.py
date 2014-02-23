#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
import ejm_rcparams
import geom

def corr_angle_theory(bins=200):
    sigmas = np.linspace(0.0, np.pi, bins + 1)[:-1]
    p = 0.5 * np.sin(sigmas)
    dsigma = np.mean(sigmas[1:] - sigmas[:-1])
    p /= (p * dsigma).sum()
    return sigmas, p

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dats', nargs='*',
        help='Data directory files')
    args = parser.parse_args()

    R_drop = 16.0
    ns = (100,)
    ns=range(20)
    print(args.dats)
    for d, n in zip(args.dats, ns):
        sigs, p, p_err = np.loadtxt(d, unpack=True)
        sigs, p_theory = corr_angle_theory(bins=len(sigs))
        p_dev = p - p_theory

        vp = 100.0 * n * 0.7 / geom.sphere_volume(R_drop, 3)
        l = r'$R=%.2g\si{\micro\metre}$, $\theta=%.2g\%%$' % (R_drop, vp)
        pp.errorbar(sigs, p_dev, yerr=p_err, label=l)

    pp.axhline(0.0, label=r'Theory', c='black')
    pp.xlabel(r'$\phi$ (Rad)')
    pp.ylabel(r'$\mathrm{P}(\phi) - \mathrm{P}_0(\phi)$')
    pp.xlim([0.0, np.pi])
    pp.legend()
    pp.show()
