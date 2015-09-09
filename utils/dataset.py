from __future__ import print_function, division
from os.path import join, dirname, normpath, basename
import glob
import pickle
import numpy as np
import scipy.stats as st
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from spatious import vector
from spatious.distance import pdist_angle


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


def res_to_bins(r_max, dr):
    return int(round((r_max) / dr))


def get_func_n(rs, x_max, dx, func):
    bins = res_to_bins(x_max, dx)
    return np.histogram(func(rs), bins=bins, range=[0.0, x_max])


def get_radial_n(rs, r_max, dr):
    return get_func_n(rs, r_max, dr, func=pdist)


def get_angle_n(rs, s_max, ds):
    return get_func_n(rs, s_max, ds, func=pdist_angle)


def get_distance_n(rs, R_max, dr):
    return get_func_n(rs, R_max, dr, func=vector.vector_mag)


def freq_to_corr(n_actual, n_actual_err,
                 n_ideal, n_ideal_err):
    '''
    Convert actual and ideal measured frequencies into a correlation function,
    assuming the histogram of both has the same range and bins.
    '''
    p_actual = n_actual / float(n_actual.sum())
    p_actual_err = n_actual_err / float(n_actual.sum())
    p_ideal = n_ideal / float(n_ideal.sum())
    p_ideal_err = n_ideal_err / float(n_ideal.sum())
    g = p_actual / p_ideal
    g_err = g * qsum(p_actual_err / p_actual, p_ideal_err / p_ideal)
    return g, g_err


def set_average(dsets):
    fs = []
    for dset in dsets:
        f, bins = dset
        fs.append(f)
    f_mean, f_err = mean_and_err(fs)
    return f_mean, f_err, bins


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
    return np.sqrt(np.sum(np.square(args), axis=0))


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


def pickle_dump(dset_paths, pickle_name):
    s = Superset(dset_paths)
    with open(pickle_name, 'wb') as f:
        pickle.dump(s, f)


def pickle_load(pickle_name):
    with open(pickle_name, 'rb') as f:
        return pickle.load(f)


class Dataset(object):
    buff = 1.1
    V_p = 0.7
    R_p = ((3.0 / 4.0) * V_p / np.pi) ** (1.0 / 3.0)
    A_p = np.pi * R_p ** 2
    v = 13.5

    def __init__(self, run_fnames, theta_max=None, force_fullsphere=False,
                 filter_z_flag=False):
        self.run_fnames = run_fnames
        self.is_hemisphere = True
        self.force_fullsphere = force_fullsphere
        self.filter_z_flag = filter_z_flag
        self.theta_max = theta_max
        self.is_hemisphere = not self.force_fullsphere

        if not run_fnames:
            raise Exception('No dyn files given')

        R_drops, set_codes = [], []
        for d in self.run_fnames:
            set_dirname = join(dirname(self.run_fnames[0]), '..')
            set_code = basename(normpath(set_dirname))
            stat = np.load('{}/static.npz'.format(set_dirname))
            R_drops.append(stat['R_d'])
            set_codes.append(set_code)

        if not all(code == set_codes[0] for code in set_codes):
            raise Exception
        self.code = set_codes[0]
        if not all(R == R_drops[0] for R in R_drops):
            raise Exception
        self.R = float(R_drops[0])

        if self.theta_max is None:
            self.theta_max = np.pi / 3.0
        if self.theta_max > np.pi / 2.0:
            raise Exception('theta_max must be less than or equal to pi / 2')

        def make_valid_func(R_drop, theta_max, force_fullsphere):
            def valid(r):
                r_mag = vector.vector_mag(r)
                if (not force_fullsphere) and r[-1] < 0.0:
                    return False
                if r_mag > R_drop:
                    return False
                theta = np.abs(np.arccos(r[-1] / r_mag))
                if theta_max < theta < (np.pi - theta_max):
                    return False
                return True
            return valid

        self.valid_func = make_valid_func(self.R, self.theta_max,
                                          self.force_fullsphere)

        self.xyzs = []
        for d in self.run_fnames:
            xyz = np.array([x for x in np.load(d)['r'] if self.valid_func(x)])
            if xyz.shape[0] > 0:
                self.xyzs.append(xyz)
        if self.filter_z_flag:
            self.filter_z()

    def get_A_drop(self):
        return A_sector(self.R, self.theta_max, self.is_hemisphere)

    def get_V_drop(self):
        return V_sector(self.R, self.theta_max, self.is_hemisphere)

    def get_n(self):
        ns = [len(xyz) for xyz in self.xyzs]
        return mean_and_err(ns)

    def get_mean(self):
        r_means = [np.mean(vector.vector_mag(xyz) / self.R)
                   for xyz in self.xyzs if len(xyz)]
        r_mean, r_mean_err = mean_and_err(r_means)
        return r_mean, r_mean_err

    def get_var(self):
        r_vars = [np.var(vector.vector_mag(xyz) / self.R, dtype=np.float64)
                  for xyz in self.xyzs if len(xyz)]
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

    def get_ns(self, dr):
        return set_average([get_distance_n(xyz, self.buff * self.R, dr)
                            for xyz in self.xyzs])

    def get_rhos(self, dr):
        ns, ns_err, R_edges = self.get_ns(dr)
        V_edges = V_sector(R_edges, self.theta_max, self.is_hemisphere)
        dVs = np.diff(V_edges)
        rhos = ns / dVs
        rhos_err = ns_err / dVs
        return rhos, rhos_err, R_edges

    def get_rhos_norm(self, dr):
        rhos, rhos_err, R_edges = self.get_rhos(dr)
        rho_0, rho_0_err = self.get_rho_0()
        rhos_norm = rhos / rho_0
        rhos_norm_err = rhos_norm * np.sqrt(np.square(rhos_err / rhos) +
                                            np.square(rho_0_err / rho_0))
        R_edges_norm = R_edges / self.R
        return rhos_norm, rhos_norm_err, R_edges_norm

    def get_i_peak(self, alg, dr):
        rho_0, rho_0_err = self.get_rho_0()
        rhos, rhos_err, R_edges = self.get_rhos(dr)

        if alg == 'mean':
            rho_base = rho_0
        elif alg == 'median':
            rho_base = np.median(rhos) + 0.2 * (np.max(rhos) - np.median(rhos))
        try:
            i_peak = line_intersections_up(rhos, rho_base)[-1]
        except IndexError:
            print(self.run_fnames[0])
            raise Exception(self.run_fnames[0])
        return i_peak

    def get_R_peak(self, alg, dr):
        i_peak = self.get_i_peak(alg, dr)
        ns, ns_err, R_edges = self.get_ns(dr)

        Rs = 0.5 * (R_edges[:-1] + R_edges[1:])
        R_peak = Rs[i_peak - 1]
        R_peak_err = (R_edges[1] - R_edges[0]) / 2.0
        return R_peak, R_peak_err

    def get_R_peak_norm(self, alg, dr):
        i_peak = self.get_i_peak(alg, dr)
        ns, ns_err, R_edges = self.get_ns(dr)

        Rs = 0.5 * (R_edges[:-1] + R_edges[1:])
        R_peak = Rs[i_peak]
        R_peak_err = (R_edges[1] - R_edges[0]) / 2.0
        return R_peak / self.R, R_peak_err / self.R

    def get_n_peak(self, alg, dr):
        ns, ns_err, R_edges = self.get_ns(dr)

        try:
            i_peak = self.get_i_peak(alg, dr)
        except Exception:
            n_peak = np.nan
            n_peak_err = np.nan
        else:
            n_peak = ns[i_peak:].sum()
            n_peak_err = np.sqrt(np.sum(np.square(ns_err[i_peak:])))
        return n_peak, n_peak_err

    def get_n_bulk(self, alg, dr):
        i_peak = self.get_i_peak(alg, dr)
        ns, ns_err, R_edges = self.get_ns(dr)

        n_bulk = ns[:i_peak].sum()
        n_bulk_err = np.sqrt(np.sum(np.square(ns_err[:i_peak])))
        return n_bulk, n_bulk_err

    def get_V_bulk(self, alg, dr):
        R_peak, R_peak_err = self.get_R_peak(alg, dr)

        V_bulk = V_sector(R_peak, self.theta_max, self.is_hemisphere)
        V_bulk_err = V_bulk * 3.0 * (R_peak_err / R_peak)
        return V_bulk, V_bulk_err

    def get_V_peak(self, alg, dr):
        V_bulk, V_bulk_err = self.get_V_bulk(alg, dr)

        V_peak = self.get_V_drop() - V_bulk
        V_peak_err = V_bulk_err
        return V_peak, V_peak_err

    def get_rho_peak(self, alg, dr):
        n_peak, n_peak_err = self.get_n_peak(alg, dr)
        V_peak, V_peak_err = self.get_V_peak(alg, dr)

        rho_peak = n_peak / V_peak
        rho_peak_err = rho_peak * qsum(n_peak_err / n_peak,
                                       V_peak_err / V_peak)
        return rho_peak, rho_peak_err

    def get_rho_peak_norm(self, alg, dr):
        rho_peak, rho_peak_err = self.get_rho_peak(alg, dr)
        rho_0, rho_0_err = self.get_rho_0()

        rho_peak_norm = rho_peak / rho_0
        rho_peak_norm_err = rho_peak_norm * qsum(rho_peak_err, rho_0_err)
        return rho_peak_norm, rho_peak_norm_err

    def get_rho_bulk(self, alg, dr):
        n_bulk, n_bulk_err = self.get_n_bulk(alg, dr)
        V_bulk, V_bulk_err = self.get_V_bulk(alg, dr)
        rho_0, rho_0_err = self.get_rho_0()

        rho_bulk = n_bulk / V_bulk
        rho_bulk_err = rho_bulk * qsum(n_bulk_err / n_bulk,
                                       V_bulk_err / V_bulk)
        return rho_bulk, rho_bulk_err

    def get_rho_bulk_norm(self, alg, dr):
        rho_bulk, rho_bulk_err = self.get_rho_bulk(alg, dr)
        rho_0, rho_0_err = self.get_rho_0()

        rho_bulk_norm = rho_bulk / rho_0
        rho_bulk_norm_err = rho_bulk_norm * qsum(rho_bulk_err, rho_0_err)
        return rho_bulk_norm, rho_bulk_norm_err

    def get_f_peak(self, alg, dr):
        n_peak, n_peak_err = self.get_n_peak(alg, dr)
        n, n_err = self.get_n()

        f_peak = n_peak / n
        f_peak_err = f_peak * qsum(n_peak_err / n_peak, n_err / n)
        return f_peak, f_peak_err

    def get_f_bulk(self, alg, dr):
        n_bulk, n_bulk_err = self.get_n_bulk(alg, dr)
        n, n_err = self.get_n()

        f_bulk = n_bulk / n
        f_bulk_err = f_bulk * qsum(n_bulk_err / n_bulk, n_err / n)
        return f_bulk, f_bulk_err

    def get_f_peak_uni(self, alg, dr):
        V_peak, V_peak_err = self.get_V_peak(alg, dr)
        f_peak_uni = V_peak / self.get_V_drop()
        f_peak_uni_err = V_peak_err / self.get_V_drop()
        return f_peak_uni, f_peak_uni_err

    def get_f_peak_excess(self, alg, dr):
        f_peak, f_peak_err = self.get_f_peak(alg, dr)
        f_peak_uni, f_peak_uni_err = self.get_f_peak_uni(alg, dr)
        f_peak_excess = f_peak - f_peak_uni
        f_peak_excess_err = qsum(f_peak_err, f_peak_uni_err)
        return f_peak_excess, f_peak_excess_err

    def get_eta(self, alg, dr):
        n_peak, n_peak_err = self.get_n_peak(alg, dr)
        eta = n_peak * self.A_p / self.get_A_drop()
        eta_err = n_peak_err * self.A_p / self.get_A_drop()
        return eta, eta_err

    def get_gamma_const(self):
        return self.R / self.v

    def get_k_const(self):
        return self.R_p * self.R / self.A_p

    def get_analytic_match(self, alg, dr, gamma, k):
        eta, eta_err = self.get_eta(alg, dr)
        eta_0, eta_0_err = self.get_eta_0()

        LHS = (1.0 - eta) * (eta_0 - eta)

        gamma_coeff = self.get_gamma_const() * eta
        c_term = gamma * gamma_coeff
        k_coeff = self.get_k_const() * eta ** 2
        b_term = k * k_coeff
        RHS = c_term + b_term
        return LHS - RHS

    def get_gamma_small_eta(self, alg, dr):
        eta, eta_err = self.get_eta(alg, dr)
        eta_0, eta_0_err = self.get_eta_0()
        LHS = (1.0 - eta) * (eta_0 - eta)

        gamma_coeff = self.get_gamma_const() * eta
        gamma = LHS / gamma_coeff
        dg_deta_0 = (self.v / self.R) * (1.0 - eta) / eta
        dg_deta = -((self.v / self.R) * (1.0 / eta) *
                    (eta_0 - 1.0 + (1.0 - eta) * (eta_0 - eta) / eta))
        gamma_err = qsum(dg_deta_0 * eta_0_err, dg_deta * eta_err)
        return gamma, gamma_err

    def get_k_fixed_gamma(self, alg, dr, gamma, gamma_err):
        eta, eta_err = self.get_eta(alg, dr)
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

    def get_xyz_ideal(self, n_samples, R_min):
        def make_valid_func(R_min):
            def valid_func(r):
                return (self.valid_func(r) and
                        vector.vector_mag_sq(r) > R_min ** 2)
            return valid_func
        valid_func = make_valid_func(R_min)

        dim = self.xyzs[0].shape[-1]

        return vector.rejection_pick(L=2.0 * self.R, n=n_samples, d=dim,
                                     valid=valid_func)

    def get_xyzs_actual(self, R_min):
        return [xyz[vector.vector_mag_sq(xyz) > R_min ** 2]
                for xyz in self.xyzs]

    def get_radial_ns_actual(self, dr, R_min):
        xyzs = self.get_xyzs_actual(R_min)
        sets = [get_radial_n(xyz, 2.0 * self.R, dr)
                for xyz in xyzs]
        return set_average(sets)

    def get_radial_ns_ideal(self, dr, n_samples, R_min):
        xyz = self.get_xyz_ideal(n_samples, R_min)
        n, R = get_radial_n(xyz, 2.0 * self.R, dr)
        n_err = np.sqrt(n)
        return n, n_err, R

    def get_rcf(self, dr, n_samples, R_min):
        na, na_err, Ra = self.get_radial_ns_actual(dr, R_min)
        ni, ni_err, Ri = self.get_radial_ns_ideal(dr, n_samples, R_min)
        g, g_err = freq_to_corr(na, na_err, ni, ni_err)
        return g, g_err, Ra

    def get_angle_ns_actual(self, ds, R_min):
        xyzs = self.get_xyzs_actual(R_min)
        sets = [get_angle_n(xyz, 2.0 * self.theta_max, ds)
                for xyz in xyzs]
        return set_average(sets)

    def get_angle_ns_ideal(self, ds, n_samples, R_min):
        xyz = self.get_xyz_ideal(n_samples, R_min)
        n, s = get_angle_n(xyz, 2.0 * self.theta_max, ds)
        n_err = np.sqrt(n)
        return n, n_err, s

    def get_acf(self, ds, n_samples, R_min):
        na, na_err, sa = self.get_angle_ns_actual(ds, R_min)
        ni, ni_err, si = self.get_angle_ns_ideal(ds, n_samples, R_min)
        g, g_err = freq_to_corr(na, na_err, ni, ni_err)
        return g, g_err, sa

    def filter_z(self, dz=2.0):
        for xyz in self.xyzs:
            xyz[:, -1] += np.random.uniform(-dz / 2.0, dz / 2.0, size=len(xyz))

    def get_direct(self):
        set_dirname = join(dirname(self.run_fnames[0]), '..')
        track = np.load('{}/tracking.npz'.format(set_dirname))
        return track['t'], track['r1'], track['r2']

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['valid_func']
        return state


def unzip(results):
    return zip(*results)


def get_dset(dset_name, *args, **kwargs):
    return Dataset(glob.glob(join(dset_name, 'dyn/*.npz')), *args, **kwargs)


class Superset(object):

    def __init__(self, dset_dirnames, theta_max=None):
        self.dset_dirnames = dset_dirnames
        self.theta_max = theta_max
        self.sets = [get_dset(d, self.theta_max) for d in self.dset_dirnames]

    def get_analytic_match(self, alg, dr, gamma, k):
        return np.array([s.get_analytic_match(alg, dr, gamma, k)
                         for s in self.sets])

    def fit_to_model(self, alg, dr):
        def f(xdata, gamma, k):
            return self.get_analytic_match(alg, dr, gamma, k)
        popt, pcov = curve_fit(f, None, np.zeros([len(self.sets)]))
        gamma_fit, k_fit = popt
        gamma_fit_err, k_fit_err = np.sqrt(np.diag(pcov))
        return gamma_fit, gamma_fit_err, k_fit, k_fit_err

    def get_mean(self):
        return unzip([d.get_mean() for d in self.sets])

    def get_var(self):
        return unzip([d.get_var() for d in self.sets])

    def get_vf(self):
        return unzip([d.get_vf() for d in self.sets])

    def get_eta_0(self):
        return unzip([d.get_eta_0() for d in self.sets])

    def get_eta(self, alg, dr):
        return unzip([d.get_eta(alg, dr) for d in self.sets])

    def get_f_peak(self, alg, dr):
        return unzip([d.get_f_peak(alg, dr) for d in self.sets])

    def get_f_peak_model(self, gamma, k):
        return np.array([d.get_f_peak_model(gamma, k) for d in self.sets])

    def get_R(self):
        return np.array([d.R for d in self.sets])
