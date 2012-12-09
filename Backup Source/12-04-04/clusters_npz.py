'''
Created on 3 Apr 2012

@author: ejm
'''

import glob

if __name__ == '__main__':
    dirname = raw_input('Enter directory: ')
    if dirname[-1] != '/': dirname += '/'
    fnames = glob.glob(dirname)
    glob.glob1(dirname, '*.npz')