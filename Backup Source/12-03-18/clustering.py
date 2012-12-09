'''
Created on 18 Mar 2012

@author: ejm
'''

import numpy as np

def cluster_params_out(datdir, params_fname, dat_fname, thresh, L):
    f = open(datdir + params_fname, 'w')
    f.write(dat_fname + '\n')
    f.write('%f\n' % thresh)
    f.write('%f %f %f\n' % 3 * (L,)) 
    f.close()

def cluster_r_out(fname, r):
    f = open(fname + '_r.dat', 'w')
    for i_x in range(r.shape[0]):
        f.write('%f' % r[i_x, 0])
        for i_y in range(1, r.shape[1]):
            f.write(' %f' % r[i_x, i_y])
        for i_y in range(r.shape[1], 3):
            f.write(' %f' % 0.0)
        f.write('\n')
    f.close()

def npz_to_plain_r(fname):
    f_npz = np.load(fname + '.npz')
    r = f_npz['r']
    f_npz.close()
    cluster_r_out(fname, r)

def main():
    import glob
    state_fnames = glob.glob('*.npz')
    for state_fname in state_fnames:
        npz_to_plain_r(state_fname)

if __name__ == '__main__':
    main()