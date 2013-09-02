#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import ejm_rcparams
import matplotlib.pyplot as pp
import yaml
import glob
import utils    
import scipy.ndimage.filters as filters
import butils

buff = 1.2

def parse_dir(dirname, s=0):
    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    dim = yaml_args['dim']
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']
    try:
        r_c = yaml_args['particle_args']['collide_args']['R']
    except KeyError:
        r_c = 0.0

    dyns = sorted(glob.glob('%s/dyn/*.npz' % dirname), key=butils.t)[::-1]

    if s == 0: pass
    elif s > len(dyns): raise Exception('Requested %i samples but only %i available' % (s, len(dyns)))
    else: dyns = dyns[:s]

    print('For dirname %s using %i samples' % (dirname, len(dyns)))
    rs = []
    for dyn in dyns:
        dyndat = np.load(dyn)
        r_head = dyndat['r']
        rs.append(utils.vector_mag(r_head))
    return np.array(rs), dim, R_drop, r_c

def make_hist(rs, R_drop, bins=100):
    ns = []
    for r in rs:
        n, R_edges = np.histogram(r, bins=bins, range=[0.0, buff * R_drop])
        ns.append(n)
    ns = np.array(ns)
    n = np.mean(ns, axis=0)
    n_err = np.std(ns, axis=0) / np.sqrt(len(ns))
    return R_edges, n, n_err

parser = argparse.ArgumentParser(description='Analyse droplet distributions')
parser.add_argument('dirs', nargs='*',
    help='Data directories')
parser.add_argument('-s', '--samples', type=int, default=0,
    help='Number of samples to use')
parser.add_argument('-b', '--bins', type=int, default=20,
    help='Number of bins to use')
parser.add_argument('-g', '--smooth', type=float, default=0.0,
    help='Length scale of gaussian blur')
parser.add_argument('-rr', '--rawr', default=False, action='store_true',
    help='Disable radial distance normalisation by droplet radius')
parser.add_argument('-rd', '--rawd', default=False, action='store_true',
    help='Disable density normalisation by mean density')
parser.add_argument('--vfprag', default=False, action='store_true',
    help='Use constant physical particle volume rather than calculated value')
parser.add_argument('-i', '--interactive', default=False, action='store_true',
    help='Show plot interactively')
args = parser.parse_args()

if args.vfprag:
    def particle_volume(*args, **kwargs):
        return 0.7
else:
    particle_volume = utils.sphere_volume

if not args.interactive: fig = pp.figure(figsize=ejm_rcparams.get_figsize(width=452, factor=0.7))
else: fig = pp.figure()
ax = fig.gca()

ax.set_ylim(0.0, 1e-6)
ax.set_xlim(0.0, 1e-6)

for dirname in args.dirs:
    rs, dim, R_drop, r_c = parse_dir(dirname, args.samples)
    Rs_edge, ns, ns_err = make_hist(rs, R_drop, args.bins)
    ns = filters.gaussian_filter1d(ns, sigma=args.smooth * float(len(ns)) / Rs_edge.max(), mode='constant', cval=0.0)

    n = rs.shape[1]
    V_particle = particle_volume(r_c, dim)
    V_drop = utils.sphere_volume(R_drop, dim)
    Vs_edge = utils.sphere_volume(Rs_edge, dim)
    dVs = Vs_edge[1:] - Vs_edge[:-1]
    rhos = ns / dVs
    rhos_err = ns_err / dVs
    rho_0 = n / V_drop
    Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])
    vf = n * V_particle / V_drop
    af = n * utils.sphere_area(r_c, dim - 1) / utils.sphere_area(R_drop, dim)

    if not args.rawd:
        rhos /= rho_0
        rhos_err /= rho_0    
    if not args.rawr:
        Rs /= R_drop

    label = r'Dir: %s, R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (dirname, R_drop, 100.0 * vf)
    ax.errorbar(Rs, rhos, yerr=rhos_err, label=label)
    ax.set_ylim(None, max(ax.get_ylim()[1], 1.1 * rhos[Rs / Rs.max() > 0.5].max()))
    ax.set_xlim(None, max(ax.get_xlim()[1], 1.1 * Rs.max()))

ax.legend(loc='upper left')
ax.set_xlabel(r'$r$' if args.rawr else r'$r / \mathrm{R}$')
ax.set_ylabel(r'$\rho(r)$' if args.rawd else r'$\rho(r) / \rho_0$')

if not args.interactive: fig.savefig('hist.pdf', bbox_inches='tight')
else: pp.show()