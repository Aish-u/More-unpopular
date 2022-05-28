# This will contain any preprocessing functions we need with the Glade catalog


#Some suggested utility functions:
'''
1. Given a RA, Dec, find the nearest galaxy in GLADE
2. Get a list of galaxies within N megaparsecs (where N is defined by the user...)
3. Given a TESS Sector, maybe find all galaxies observed within that sector? - instead we'll 
pull RA and DEC of each galaxy and check for transients ?

'''

import numpy as np
import pandas as pd

from astroquery.mast import Tesscut
from astroquery.mast.utils import parse_input_location
import os
from astropy.coordinates import SkyCoord
import tess_cpm




'''
def load_Glade(fpath):
    data = np.loadtxt(fpath,dtype=str,usecols=(0,1))
    return data
    
data = load_Glade('../../GLADE+.txt')
'''


glade = pd.read_csv("glade(13).txt",sep =" ", names=["Glade_no","PGC_no","GWGCName","HyperLedaName","2MASSName","WISExSCOSNAME","SDSS-DR16QName","Object-typeFlag","RA","Dec","B","B_err","B_Flag","B_Abs","J","J_err","H","H_err","K","K_err","W1","W1_err","W2","W2_err","W1_flag","B_J","B_J_err","z_helio","z_cmb","z_flag","v_err","z_err","d_L","d_L_err","dist_Flag","M_mass","M_mass_err","Merger_rate","Merger_rate_err"])
g = glade.sort_values(by="d_L",ascending=True)  # sorts in terms of distance d_L closest to furthest
cols = [0,1,8,9,32,33]
g = g[g.columns[cols]]



#change type of RA and DEC from str to float
Ra = g['RA'].to_list()
Dec = g['Dec'].to_list()
#writes a new ordered file to look at if needed. remove '#' from the code below to have it write a txt file
#g.to_csv(r'Ordered_glade.txt', header=None, index=None, sep='\t', mode='a')



mypath = './fits_files/'
def check_before_download(coordinates=None, size=5, sector=None, path=mypath, inflate=True, objectname=None, force_download=False):
    coords = parse_input_location(coordinates, objectname)
    ra = f"{coords.ra.value:.6f}"
    matched = [path+m for m in os.listdir(path) if ra in m]
    if (len(matched) != 0) and (force_download == False):
        print(f"Found the following FITS files in the \"{path}/\" directory with matching RA values.")
        print(matched)
        print("If you still want to download the file, set the \'force_download\' keyword to True.")
        return matched
    else:
        path_to_FFIs = Tesscut.download_cutouts(coordinates=coordinates, size=size, sector=sector, path=path, inflate=inflate, objectname=objectname)
        print(path_to_FFIs)
        return path_to_FFIs
# We need to write code to automatically try different coordinates
# Write a flag to decide if you want to generate plots


# this loop creates fits files iterating through all the galaxies in glade.txt
for i,j  in zip(Ra,Dec):
  ra=i
  dec=j

  path_to_FFIs=check_before_download(coordinates=SkyCoord(ra,dec,unit="deg"),size = 50)
  path_to_FFIs=check_before_download(coordinates=SkyCoord(ra,dec,unit="deg"),size = 50)
  s1= tess_cpm.Source(path_to_FFIs[0],remove_bad=True)















