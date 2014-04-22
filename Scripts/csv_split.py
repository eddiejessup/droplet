import numpy as np
import glob
import os
import shutil

dirname = '/Users/ejm/Projects/Bannock/Data/drop/exp/xyz_filt_split'

try:
    os.mkdir(dirname)
except OSError:
    pass

for fname in glob.glob('/Users/ejm/Projects/Bannock/Data/drop/exp/xyz_filt/*.csv'):
    stats = os.path.basename(fname).split('_')
    code = stats[0]
    newname = os.path.basename(fname)
    newdir = os.path.join(dirname, code)
    try:
        os.mkdir(newdir)
    except OSError:
        pass
    newpath = os.path.join(newdir, newname)
    print newpath
    shutil.copy(fname, newpath)
