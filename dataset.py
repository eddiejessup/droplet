#! /usr/bin/env python3

from os.path import join, dirname, normpath, basename
import numpy as np
from ciabatta import vector
import scipy.stats as st
from scipy.optimize import curve_fit
import glob
import operator


def f_peak_model(eta_0, R, k, gamma,
                 R_p=None, v=None):
    if R_p is None:
        R_p = Dataset.R_p
    if v is None:
        v = Dataset.v
    A_p = np.pi * R_p ** 2
    k_const = R_p * R / A_p
    gamma_const = R / v
    r = np.roots([eta_0 * (1.0 - k_const * k),
                  -1.0 - eta_0 - gamma_const * gamma,
                  1.0])
    return r[-1]


def mean_and_err(x):
    '''
    Return the mean and standard error on the mean of a set of values.

    Parameters
    ----------
    x: array shape (n, ...)
        `n` sample values or sets of sample values.

    Returns
    -------
    mean: array, shape (...)
        The mean of `x` over its first axis.
        The other axes are left untouched.
    stderr: array, shape (...)
        The standard error on the mean of `x` over its first axis.
    '''
    return np.mean(x, axis=0), st.sem(x, axis=0)


def qsum(*args):
    if len(args) == 1:
        return np.sqrt(np.sum(np.square(args[0])))
    return np.sqrt(np.sum(np.square(args)))


def V_sector(R, theta, hemisphere=False):
    '''
    Volume of two spherical sectors with half cone angle theta.
    Two is because we consider two sectors, either side of the sphere's centre.
    See en.wikipedia.org/wiki/Spherical_sector, where its phi == this theta.
    '''
    if theta > np.pi / 2.0:
        raise Exception('Require sector half-angle at most pi / 2')
    V_sector = 2.0 * (2.0 / 3.0) * np.pi * R ** 3 * (1.0 - np.cos(theta))
    if hemisphere:
        V_sector /= 2.0
    return V_sector


def A_sector(R, theta, hemisphere=False):
    '''
    Surface area of two spherical sectors with half cone angle theta.
    Two is because we consider two sectors, either side of the sphere's centre.
    See en.wikipedia.org/wiki/Spherical_sector, where its phi == this theta.
    '''
    if theta > np.pi / 2.0:
        raise Exception('Require sector half-angle at most pi / 2')
    A_sector = 2.0 * 2.0 * np.pi * R ** 2 * (1.0 - np.cos(theta))
    if hemisphere:
        A_sector /= 2.0
    return A_sector


def line_intersections_up(y, y0):
    inds = []
    for i in range(len(y) - 1):
        if y[i] <= y0 and y[i + 1] > y0:
            inds.append(i + 1)
    return inds


class Dataset(object):
    buff = 1.1
    V_p = 0.7
    R_p = ((3.0 / 4.0) * V_p / np.pi) ** (1.0 / 3.0)
    A_p = np.pi * R_p ** 2
    v = 13.5

    def __init__(self, run_fnames, theta_max=None, force_hemisphere=False):
        self.run_fnames = run_fnames

        R_drops, is_hemispheres, set_codes = [], [], []
        for d in self.run_fnames:
            set_dirname = join(dirname(self.run_fnames[0]), '..')
            set_code = basename(normpath(set_dirname))
            stat = np.load('%s/static.npz' % set_dirname)
            R_drops.append(stat['R_d'])
            if 'hemisphere' not in stat:
                is_hemisphere = False
            else:
                is_hemisphere = stat['hemisphere']
            is_hemispheres.append(is_hemisphere)
            set_codes.append(set_code)

        if not all(code == set_codes[0] for code in set_codes):
            raise Exception
        self.code = set_codes[0]
        if not all(R == R_drops[0] for R in R_drops):
            raise Exception
        self.R = float(R_drops[0])
        if not all(h == is_hemispheres[0] for h in is_hemispheres):
            raise Exception
        self.is_hemisphere = is_hemispheres[0]

        if force_hemisphere:
            if self.is_hemisphere:
                raise Exception('Forcing hemisphere data that already is')
            else:
                self.is_hemisphere = True

        if theta_max is None:
            if self.is_hemisphere:
                self.theta_max = np.pi / 3.0
            else:
                self.theta_max = np.pi / 2.0
        else:
            self.theta_max = theta_max

        self.rs = []
        for d in self.run_fnames:
            xyz = np.load(d)['r']
            r = vector.vector_mag(xyz)
            theta = np.arccos(xyz[..., -1] / r)

            if force_hemisphere:
                hemisphere_force = theta < np.pi / 2.0
                xyz = xyz[hemisphere_force]
                r = r[hemisphere_force]
                theta = theta[hemisphere_force]
                self.hemisphere = True

            if self.is_hemisphere:
                if np.any(theta > 2.0 * np.pi / 2.0):
                    raise Exception(len(theta[theta > np.pi / 2.0]) /
                                    len(theta))
            # else:
                # if not np.any(theta > 1.1 * np.pi / 2.0):
                #     raise Exception('Are you sure this is not hemisphere '
                #                     'data?')
            valid = np.logical_or(np.abs(theta) < self.theta_max,
                                  np.abs(theta) > (np.pi - self.theta_max))
            xyz = xyz[valid]
            self.rs.append(vector.vector_mag(xyz))

    def get_A_drop(self):
        return A_sector(self.R, self.theta_max, self.is_hemisphere)

    def get_V_drop(self):
        return V_sector(self.R, self.theta_max, self.is_hemisphere)

    def get_n(self):
        ns = [len(r) for r in self.rs]
        return mean_and_err(ns)

    def get_mean(self):
        r_means = [np.mean(r / self.R) for r in self.rs if len(r)]
        r_mean, r_mean_err = mean_and_err(r_means)
        return r_mean, r_mean_err

    def get_var(self):
        r_vars = [np.var(r / self.R,
                         dtype=np.float64) for r in self.rs if len(r)]
        r_var, r_var_err = mean_and_err(r_vars)
        return r_var, r_var_err

    def get_vf(self):
        n, n_err = self.get_n()

        vf = n * self.V_p / self.get_V_drop()
        vf_err = n_err * self.V_p / self.get_V_drop()
        return vf, vf_err

    def get_vp(self):
        vf, vf_err = self.get_vf()
        return 100.0 * vf, 100.0 * vf_err

    def get_eta_0(self):
        n, n_err = self.get_n()

        eta_0 = n * self.A_p / self.get_A_drop()
        eta_0_err = n_err * self.A_p / self.get_A_drop()
        return eta_0, eta_0_err

    def get_rho_0(self):
        n, n_err = self.get_n()

        rho_0 = n / self.get_V_drop()
        rho_0_err = n_err / self.get_V_drop()
        return rho_0, rho_0_err

    def get_ns(self, res):
        '''
        Returns
        -------
        ns
        ns_err
        R_edges
        '''
        bins = int(round((self.buff * self.R) / res))
        nss = []
        for r in self.rs:
            ns, R_edges = np.histogram(r, bins=bins,
                                       range=[0.0, self.buff * self.R])
            nss.append(ns)
        ns, ns_err = mean_and_err(nss)
        return ns, ns_err, R_edges

    def get_rhos(self, res):
        ns, ns_err, R_edges = self.get_ns(res)
        V_edges = V_sector(R_edges, self.theta_max, self.is_hemisphere)
        dVs = np.diff(V_edges)
        rhos = ns / dVs
        rhos_err = ns_err / dVs
        return rhos, rhos_err, R_edges

    def get_rhos_norm(self, res):
        rhos, rhos_err, R_edges = self.get_rhos(res)
        rho_0, rho_0_err = self.get_rho_0()
        rhos_norm = rhos / rho_0
        rhos_norm_err = rhos_norm * np.sqrt(np.square(rhos_err / rhos) +
                                            np.square(rho_0_err / rho_0))
        R_edges_norm = R_edges / self.R
        return rhos_norm, rhos_norm_err, R_edges_norm

    def get_i_peak(self, alg, res):
        rho_0, rho_0_err = self.get_rho_0()
        rhos, rhos_err, R_edges = self.get_rhos(res)

        if alg == 'mean':
            rho_base = rho_0
        elif alg == 'median':
            rho_base = np.median(rhos) + 0.2 * (np.max(rhos) - np.median(rhos))
        try:
            i_peak = line_intersections_up(rhos, rho_base)[-1]
        except IndexError:
            raise Exception(self.run_fnames[0])
        return i_peak

    def get_R_peak(self, alg, res):
        i_peak = self.get_i_peak(alg, res)
        ns, ns_err, R_edges = self.get_ns(res)

        Rs = 0.5 * (R_edges[:-1] + R_edges[1:])
        R_peak = Rs[i_peak]
        R_peak_err = (R_edges[1] - R_edges[0]) / 2.0
        return R_peak, R_peak_err

    def get_R_peak_norm(self, alg, res):
        i_peak = self.get_i_peak(alg, res)
        ns, ns_err, R_edges = self.get_ns(res)

        Rs = 0.5 * (R_edges[:-1] + R_edges[1:])
        R_peak = Rs[i_peak]
        R_peak_err = (R_edges[1] - R_edges[0]) / 2.0
        return R_peak / self.R, R_peak_err / self.R

    def get_n_peak(self, alg, res):
        i_peak = self.get_i_peak(alg, res)
        ns, ns_err, R_edges = self.get_ns(res)

        n_peak = ns[i_peak:].sum()
        n_peak_err = np.sqrt(np.sum(np.square(ns_err[i_peak:])))
        return n_peak, n_peak_err

    def get_n_bulk(self, alg, res):
        i_peak = self.get_i_peak(alg, res)
        ns, ns_err, R_edges = self.get_ns(res)

        n_bulk = ns[:i_peak].sum()
        n_bulk_err = np.sqrt(np.sum(np.square(ns_err[:i_peak])))
        return n_bulk, n_bulk_err

    def get_V_bulk(self, alg, res):
        R_peak, R_peak_err = self.get_R_peak(alg, res)

        V_bulk = V_sector(R_peak, self.theta_max, self.is_hemisphere)
        V_bulk_err = V_bulk * 3.0 * (R_peak_err / R_peak)
        return V_bulk, V_bulk_err

    def get_V_peak(self, alg, res):
        V_bulk, V_bulk_err = self.get_V_bulk(alg, res)

        V_peak = self.get_V_drop() - V_bulk
        V_peak_err = V_bulk_err
        return V_peak, V_peak_err

    def get_rho_peak(self, alg, res):
        n_peak, n_peak_err = self.get_n_peak(alg, res)
        V_peak, V_peak_err = self.get_V_peak(alg, res)

        rho_peak = n_peak / V_peak
        rho_peak_err = rho_peak * qsum(n_peak_err / n_peak,
                                       V_peak_err / V_peak)
        return rho_peak, rho_peak_err

    def get_rho_peak_norm(self, alg, res):
        rho_peak, rho_peak_err = self.get_rho_peak(alg, res)
        rho_0, rho_0_err = self.get_rho_0()

        rho_peak_norm = rho_peak / rho_0
        rho_peak_norm_err = rho_peak_norm * qsum(rho_peak_err, rho_0_err)
        return rho_peak_norm, rho_peak_norm_err

    def get_rho_bulk(self, alg, res):
        n_bulk, n_bulk_err = self.get_n_bulk(alg, res)
        V_bulk, V_bulk_err = self.get_V_bulk(alg, res)
        rho_0, rho_0_err = self.get_rho_0()

        rho_bulk = n_bulk / V_bulk
        rho_bulk_err = rho_bulk * qsum(n_bulk_err / n_bulk,
                                       V_bulk_err / V_bulk)
        return rho_bulk, rho_bulk_err

    def get_rho_bulk_norm(self, alg, res):
        rho_bulk, rho_bulk_err = self.get_rho_bulk(alg, res)
        rho_0, rho_0_err = self.get_rho_0()

        rho_bulk_norm = rho_bulk / rho_0
        rho_bulk_norm_err = rho_bulk_norm * qsum(rho_bulk_err, rho_0_err)
        return rho_bulk_norm, rho_bulk_norm_err

    def get_f_peak(self, alg, res):
        n_peak, n_peak_err = self.get_n_peak(alg, res)
        n, n_err = self.get_n()

        f_peak = n_peak / n
        f_peak_err = f_peak * qsum(n_peak_err / n_peak, n_err / n)
        return f_peak, f_peak_err

    def get_f_bulk(self, alg, res):
        n_bulk, n_bulk_err = self.get_n_bulk(alg, res)
        n, n_err = self.get_n()

        f_bulk = n_bulk / n
        f_bulk_err = f_bulk * qsum(n_bulk_err / n_bulk, n_err / n)
        return f_bulk, f_bulk_err

    def get_f_peak_uni(self, alg, res):
        V_peak, V_peak_err = self.get_V_peak(alg, res)
        f_peak_uni = V_peak / self.get_V_drop()
        f_peak_uni_err = V_peak_err / self.get_V_drop()
        return f_peak_uni, f_peak_uni_err

    def get_f_peak_excess(self, alg, res):
        f_peak, f_peak_err = self.get_f_peak(alg, res)
        f_peak_uni, f_peak_uni_err = self.get_f_peak_uni(alg, res)
        f_peak_excess = f_peak - f_peak_uni
        f_peak_excess_err = qsum(f_peak_err, f_peak_uni_err)
        return f_peak_excess, f_peak_excess_err

    def get_eta(self, alg, res):
        n_peak, n_peak_err = self.get_n_peak(alg, res)
        eta = n_peak * self.A_p / self.get_A_drop()
        eta_err = n_peak_err * self.A_p / self.get_A_drop()
        return eta, eta_err

    def get_gamma_const(self):
        return self.R / self.v

    def get_k_const(self):
        return self.R_p * self.R / self.A_p

    def get_analytic_match(self, alg, res, gamma, k):
        eta, eta_err = self.get_eta(alg, res)
        eta_0, eta_0_err = self.get_eta_0()

        LHS = (1.0 - eta) * (eta_0 - eta)

        gamma_coeff = self.get_gamma_const() * eta
        c_term = gamma * gamma_coeff
        k_coeff = self.get_k_const() * eta ** 2
        b_term = k * k_coeff
        RHS = c_term + b_term
        return LHS - RHS

    def get_gamma_small_eta(self, alg, res):
        eta, eta_err = self.get_eta(alg, res)
        eta_0, eta_0_err = self.get_eta_0()
        LHS = (1.0 - eta) * (eta_0 - eta)

        gamma_coeff = self.get_gamma_const() * eta
        gamma = LHS / gamma_coeff
        dg_deta_0 = (self.v / self.R) * (1.0 - eta) / eta
        dg_deta = -((self.v / self.R) * (1.0 / eta) *
                    (eta_0 - 1.0 + (1.0 - eta) * (eta_0 - eta) / eta))
        gamma_err = qsum(dg_deta_0 * eta_0_err, dg_deta * eta_err)
        return gamma, gamma_err

    def get_k_fixed_gamma(self, alg, res, gamma, gamma_err):
        eta, eta_err = self.get_eta(alg, res)
        eta_0, eta_0_err = self.get_eta_0()
        LHS = (1.0 - eta) * (eta_0 - eta)

        gamma_coeff = self.get_gamma_const() * eta
        c_term = gamma * gamma_coeff

        k_coeff = self.get_k_const() * eta ** 2
        k = (LHS - c_term) / k_coeff

        dk_deta_0 = np.nan
        dk_deta = np.nan
        k_err = qsum(dk_deta_0 * eta_0_err, dk_deta * eta_err)
        return k, k_err

    def get_f_peak_model(self, gamma, k):
        eta_0, eta_0_err = self.get_eta_0()

        r = np.roots([eta_0 * (1.0 - self.get_k_const() * k),
                      -1.0 - eta_0 - self.get_gamma_const() * gamma,
                      1.0])
        return r[1]


class Superset(object):

    def __init__(self, dset_dirnames, theta_max):
        self.dset_dirnames = dset_dirnames
        self.theta_max = theta_max

        self.sets = []
        for dset_dirname in self.dset_dirnames:
            run_dirnames = glob.glob('{}/dyn/*.npz'.format(dset_dirname))
            if not run_dirnames:
                raise Exception('{}: no runs found'.format(dset_dirname))
            self.sets.append(Dataset(run_dirnames, self.theta_max))

    def map(self, f, args=[], kwargs={}):
        zipped = map(operator.methodcaller(f, *args, **kwargs), self.sets)
        unzipped = list(zip(*zipped))
        return tuple([np.array(u) for u in unzipped])

    def get_n(self):
        return self.map('get_n')

    def get_mean(self):
        return self.map('get_mean')

    def get_var(self):
        return self.map('get_var')

    def get_ns(self, res):
        return self.map('get_ns', (res,))

    def get_rhos(self, res):
        return self.map('get_rhos', (res,))

    def get_rhos_norm(self, res):
        return self.map('get_rhos_norm', (res,))

    def get_rho_0(self):
        return self.map('get_rho_0')

    def get_A_drop(self):
        return self.map('get_A_drop')

    def get_R(self):
        return np.array([s.R for s in self.sets])

    def get_code(self):
        return np.array([s.code for s in self.sets])

    def get_vf(self):
        return self.map('get_vf')

    def get_vp(self):
        return self.map('get_vp')

    def get_eta_0(self):
        return self.map('get_eta_0')

    def get_R_peak(self, alg, res):
        return self.map('get_R_peak', (alg, res))

    def get_n_peak(self, alg, res):
        return self.map('get_n_peak', (alg, res))

    def get_n_bulk(self, alg, res):
        return self.map('get_n_bulk', (alg, res))

    def get_V_bulk(self, alg, res):
        return self.map('get_V_bulk', (alg, res))

    def get_V_peak(self, alg, res):
        return self.map('get_V_peak', (alg, res))

    def get_rho_peak(self, alg, res):
        return self.map('get_rho_peak', (alg, res))

    def get_rho_peak_norm(self, alg, res):
        return self.map('get_rho_peak_norm', (alg, res))

    def get_rho_bulk(self, alg, res):
        return self.map('get_rho_bulk', (alg, res))

    def get_rho_bulk_norm(self, alg, res):
        return self.map('get_rho_bulk_norm', (alg, res))

    def get_f_peak(self, alg, res):
        return self.map('get_f_peak', (alg, res))

    def get_f_bulk(self, alg, res):
        return self.map('get_f_bulk', (alg, res))

    def get_f_peak_uni(self, alg, res):
        return self.map('get_f_peak_uni', (alg, res))

    def get_f_peak_excess(self, alg, res):
        return self.map('get_f_peak_excess', (alg, res))

    def get_eta(self, alg, res):
        return self.map('get_eta', (alg, res))

    def get_analytic_match(self, alg, res, gamma, k):
        return np.array([s.get_analytic_match(alg, res, gamma, k)
                         for s in self.sets])

    def get_gamma_small_eta(self, alg, res):
        return self.map('get_gamma_small_eta', (alg, res))

    def get_k_fixed_gamma(self, alg, res, gamma, gamma_err):
        return self.map('get_k_fixed_gamma', (alg, res, gamma, gamma_err))

    def get_f_peak_model(self, gamma, k):
        return np.array([s.get_f_peak_model(gamma, k)
                         for s in self.sets])

    def fit_to_model(self, alg, res):
        def f(xdata, gamma, k):
            return self.get_analytic_match(alg, res, gamma, k)
        popt, pcov = curve_fit(f, None, np.zeros([len(self.sets)]))
        gamma_fit, k_fit = popt
        gamma_fit_err, k_fit_err = np.sqrt(np.diag(pcov))
        return gamma_fit, gamma_fit_err, k_fit, k_fit_err
