import pickle
import os
import numpy as np
import obstructions

def get_stat(dirname):
    print('%s/cp.pkl' % dirname)
    if os.path.exists('%s/cp.pkl' % dirname):

        env = pickle.load(open('%s/cp.pkl' % dirname, 'rb'))
        stat = {'L': env.o.L,
                'r_0': env.p.r_0}
        if isinstance(env.o, obstructions.Walls):
            stat['o'] = env.o.a
        elif isinstance(env.o, obstructions.Droplet):
            stat['R'] = env.o.R
        elif isinstance(env.o, obstructions.Porous):
            stat['r'] = env.o.r
            stat['R'] = env.o.R
        return stat
    else:
        return np.load('%s/static.npz' % dirname)