# -*- coding: utf-8 -*-
'''
Created on Mon Apr 30 21:21:33 2012

@author: ejm
'''

import numpy as np
import _cluster_list_py
import utils

def get_inds(r, R_cut, L):
    link_list = _cluster_list_py.cluster_list(r, R_cut, L)
    assert r.shape[0] == link_list.shape[0]
    link_list -= 1
    used = np.zeros(link_list.shape, dtype=np.bool)
    inds = []
    for i_0 in range(len(link_list)):
        if not used[i_0]:
            inds.append([])
            i_cur = i_0
            while not used[i_cur]:
                used[i_cur] = True
                inds[-1].append(i_cur)            
                i_cur = link_list[i_cur]
    return inds

def get_pops(inds):
    return np.array([len(inds[i]) for i in range(len(inds))], dtype=np.float)

def get_rms(inds, r, L):
    rms_list = [utils.rms_com(r[inds[i]], True, L) for i in range(len(inds))]
    return np.array(rms_list)