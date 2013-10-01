#! /usr/bin/env python

from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import yaml
import glob
import utils
import geom
import scipy.ndimage.filters as filters
import scipy.stats as st
import butils

buff = 1.2

def parse_dir(dirname, s=0):
    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    dim = yaml_args['dim']
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']
    r_c = yaml_args['particle_args']['collide_args']['R']
    lu_c = yaml_args['particle_args']['collide_args']['lu']
    ld_c = yaml_args['particle_args']['collide_args']['ld']
    L_c = lu_c + ld_c

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
    return np.array(rs), dim, R_drop, r_c, L_c

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
    def particle_volume(R, L, dim):
        return geom.sphere_volume(R, dim) + (np.pi * R ** 2) * L

if not args.interactive:
    import ejm_rcparams

if not args.interactive: 
    fig = pp.figure(figsize=ejm_rcparams.get_figsize(width=452, factor=0.7))
else: fig = pp.figure()
ax = fig.gca()
ax.set_ylim(0.0, 1e-6)
ax.set_xlim(0.0, 1e-6)

vfs, afs = [], []
r_means, r_vars, r_skews, r_kurts = [], [], [], []
rho_peaks, rho_bulks = [], []
n_tots, n_peaks = [], []

for dirname in args.dirs:
    rs, dim, R_drop, r_c, L_c = parse_dir(dirname, args.samples)
    Rs_edge, ns, ns_err = make_hist(rs, R_drop, args.bins)
    ns = filters.gaussian_filter1d(ns, sigma=args.smooth * float(len(ns)) / Rs_edge.max(), mode='constant', cval=0.0)

    n = rs.shape[1]
    V_particle = particle_volume(r_c, L_c, dim)
    V_drop = geom.sphere_volume(R_drop, dim)
    Vs_edge = geom.sphere_volume(Rs_edge, dim)
    dVs = Vs_edge[1:] - Vs_edge[:-1]
    rhos = ns / dVs
    rhos_err = ns_err / dVs
    rho_0 = n / V_drop
    Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])
    vf = n * V_particle / V_drop
    af = n * geom.sphere_area(r_c, dim - 1) / geom.sphere_area(R_drop, dim)

    r_mean = np.mean(rs / R_drop)
    r_var = np.var(rs / R_drop)
    r_skew = st.skew(rs / R_drop, axis=None)
    r_kurt = st.kurtosis(rs / R_drop, axis=None)

    vfs.append(vf)
    afs.append(af)

    r_means.append(r_mean)
    r_vars.append(r_var)
    r_skews.append(r_skew)
    r_kurts.append(r_kurt)

    rho_peak = rhos[Rs > 0.5 * R_drop].max()
    i_peak = np.where(rhos == rho_peak)[0][0]
    rho_bulk = np.mean(rhos[:i_peak]) / rho_0
    rho_peak = np.mean(rhos[i_peak]) / rho_0
    rho_peaks.append(rho_peak)
    rho_bulks.append(rho_bulk)

    n_peak = ns[i_peak:].sum()
    n_peaks.append(n_peak)
    n_tots.append(n)

    if not args.rawd:
        rhos /= rho_0
        rhos_err /= rho_0    
    if not args.rawr:
        Rs /= R_drop

    if args.interactive:
        label = r'Dir: %s, R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (dirname, R_drop, 100.0 * vf)
    else:
        label = r'R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (R_drop, 100.0 * vf)
    ax.errorbar(Rs, rhos, yerr=rhos_err, label=label)
    ax.set_ylim(None, max(ax.get_ylim()[1], 1.1 * rhos[Rs / Rs.max() > 0.5].max()))
    ax.set_xlim(None, max(ax.get_xlim()[1], 1.1 * Rs.max()))

# ax.legend(loc='upper left')
# ax.set_xlabel(r'$r$' if args.rawr else r'$r / \mathrm{R}$')
# ax.set_ylabel(r'$\rho(r)$' if args.rawd else r'$\rho(r) / \rho_0$')
# if not args.interactive: fig.savefig('hist.pdf', bbox_inches='tight')
# else: pp.show()

vps = 100.0 * np.array(vfs)

# if not args.interactive: fig = pp.figure(figsize=ejm_rcparams.get_figsize(width=452, factor=0.7))
# else: fig = pp.figure()
# ax = fig.gca()
# ax.scatter(vps, r_means)
# ax.set_xlabel(r'Volume fraction (\%)')
# ax.set_ylabel(r'$<r / R>$')
# ax.set_xlim(0.0, None)
# ax.set_ylim(0.0, None)
# ax.axhline(dim / (dim + 1.0), label='${<r / R>}_0$')
# ax.legend()
# if not args.interactive: fig.savefig('r_mean.pdf', bbox_inches='tight')
# else: pp.show()

# if not args.interactive: fig = pp.figure(figsize=ejm_rcparams.get_figsize(width=452, factor=0.7))
# else: fig = pp.figure()
# ax = fig.gca()
# ax.scatter(vps, r_vars)
# ax.set_xlabel(r'Volume fraction (\%)')
# ax.set_ylabel(r'$<(r / R)^2>$')
# ax.set_xlim(0.0, None)
# ax.set_ylim(0.0, None)
# ax.axhline(dim * (1.0 / (dim + 2.0) - dim / (dim + 1.0) ** 2), label='${<(r / R)^2>}_0$')
# ax.legend()
# if not args.interactive: fig.savefig('r_var.pdf', bbox_inches='tight')
# else: pp.show()

# if not args.interactive: fig = pp.figure(figsize=ejm_rcparams.get_figsize(width=452, factor=0.7))
# else: fig = pp.figure()
# ax = fig.gca()
# ax.scatter(vps, rho_peaks, c='red', label='Peak')
# ax.scatter(vps, rho_bulks, c='blue', label='Bulk')
# ax.set_xlabel(r'Volume fraction (\%)')
# ax.set_ylabel(r'$\rho / \rho_0$')
# ax.set_xlim(0.0, None)
# ax.axhline(1.0, label=r'$\rho_0$')
# ax.legend(loc='upper right')
# if not args.interactive: fig.savefig('peak.pdf', bbox_inches='tight')
# else: pp.show()

if not args.interactive: fig = pp.figure(figsize=ejm_rcparams.get_figsize(width=452, factor=0.7))
else: fig = pp.figure()
ax = fig.gca()
ax.scatter(n_tots, n_peaks)
ax.set_xlabel(r'Number of bacteria')
ax.set_ylabel(r'Number of bacteria in the peak')
ax.set_xlim(0.0, None)
ax.set_ylim(0.0, None)
if not args.interactive: fig.savefig('n.pdf', bbox_inches='tight')
else: pp.show()

np.savetxt('n.csv', zip(n_tots, n_peaks), header='n_total n_peak')
np.savetxt('peak.csv', zip(vps, rho_peaks, rho_bulks), header='volume_percentage rho_peak rho_bulk')
np.savetxt('r_mean.csv', zip(vps, r_means, r_vars), header='volume_percentage r_mean r_variance')
