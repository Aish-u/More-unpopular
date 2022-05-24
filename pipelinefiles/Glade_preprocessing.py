# This will contain any preprocessing functions we need with the Glade catalog


#Some suggested utility functions:
'''
1. Given a RA, Dec, find the nearest galaxy in GLADE
2. Get a list of galaxies within N megaparsecs (where N is defined by the user...)
3. Given a TESS Sector, maybe find all galaxies observed within that sector?
'''

import numpy as np


def load_Glade(fpath):
    data = np.loadtxt(fpath,dtype=str,usecols=(0,1))
    return data
    
data = load_Glade('../../GLADE+.txt')