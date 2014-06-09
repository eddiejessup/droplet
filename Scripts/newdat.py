import numpy as np
import glob
import os
import droplyse

inpref = '/Users/ejm/Projects/Bannock/Data/drop/exp/xyz'
outpref = '/Users/ejm/datnew'

for d in glob.glob(os.path.join(inpref, 'D*')):
    try:
        R_drop = droplyse.code_to_param(d, exp=True)
    except Exception:
        ignores = ['118', '119', '121', '124', '223', '231', '310', '311']
        for ignore in ignores:
            if ignore in d:
                break
        else:
            raise Exception

    outdir = os.path.join(outpref, os.path.basename(d))
    os.makedirs(outdir)
    outstat = os.path.join(outdir, 'static.npz')
    np.savez(outstat, R_d=R_drop, hemisphere=True)
    outdyndir = os.path.join(outdir, 'dyn')
    os.makedirs(outdyndir)
    for f in glob.glob(os.path.join(d, '*.csv')):
        xyz = np.loadtxt(f)
        assert xyz.ndim == 2
        assert xyz.shape[-1] == 3
        outdyn = os.path.join(outdyndir, os.path.splitext(os.path.basename(f))[0] + '.npz')
        np.savez(outdyn, r=xyz)
