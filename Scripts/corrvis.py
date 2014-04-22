#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
import ejm_rcparams
import geom
import utils
import corr


def corr_angle_theory(bins=200):
    sigmas = np.linspace(0.0, np.pi, bins + 1)[:-1]
    p = 0.5 * np.sin(sigmas)
    dsigma = np.mean(sigmas[1:] - sigmas[:-1])
    p /= (p * dsigma).sum()
    return p, sigmas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dats', nargs='*',
                        help='Data directory files')
    args = parser.parse_args()

    ls = [r'Simulation, $\phi=0.2\%$, $R=\SI{16}{\micro\m}$',
          r'Experiment, droplet D22, $\phi=0.35\%$, $R=\SI{17.6}{\micro\m}$'] * 2
    for d, l in zip(args.dats, ls):
        sigs, p, p_err = np.loadtxt(d, unpack=True)

        # p_theory, sigs = corr_angle_theory(bins=len(sigs))
        r = utils.sphere_pick(3, n=1000)
        r[:, -1] = np.abs(r[:, -1])
        p_theory, sigs_theory = np.histogram(
            corr.pdist_angle(r), bins=len(sigs), density=True, range=(0.0, np.pi))
        # pp.plot(sigs_theory[:-1], p, label='Theory')

        p_dev = p - p_theory

        l = l
        l = l.replace('_', '\_')
        pp.errorbar(sigs, p_dev, yerr=p_err, label=l)
        # pp.errorbar(sigs, p, yerr=p_err, label=l)


    # pp.axhline(0.0, label=r'Theory', c='black')
    pp.xlabel(r'$\phi$ (Rad)')
    pp.ylabel(r'$\mathrm{P}(\phi) - \mathrm{P}_0(\phi)$')
    pp.xlim([0.0, np.pi])
    pp.legend()
    pp.title('Deviation of angle correlation function from that expected of a uniform distribution')
    pp.show()
