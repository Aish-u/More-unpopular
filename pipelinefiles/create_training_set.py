from astroquery.mast import Tesscut
from astroquery.mast.utils import parse_input_location
import os
from astropy.coordinates import SkyCoord
import tess_cpm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




glade = pd.read_csv("glade(13).txt",sep =" ", names=["Glade_no","PGC_no","GWGCName","HyperLedaName","2MASSName","WISExSCOSNAME","SDSS-DR16QName","Object-typeFlag","RA","Dec","B","B_err","B_Flag","B_Abs","J","J_err","H","H_err","K","K_err","W1","W1_err","W2","W2_err","W1_flag","B_J","B_J_err","z_helio","z_cmb","z_flag","v_err","z_err","d_L","d_L_err","dist_Flag","M_mass","M_mass_err","Merger_rate","Merger_rate_err"])
g = glade.sort_values(by="d_L",ascending=True)  # sorts in terms of distance d_L closest to furthest
cols = [0,1,8,9,32,33]
g = g[g.columns[cols]]

g = g.sample(n=2)

#change type of RA and DEC from str to float
Ra = g['RA'].to_list()
Dec = g['Dec'].to_list()
#writes a new ordered file to look at if needed. remove '#' from the code below to have it write a txt file
#g.to_csv(r'Ordered_glade.txt', header=None, index=None, sep='\t', mode='a')

# prints the random sample.
print(g) 

#Set the path to put the giant fits files in - DONT put these on github!
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
#we want to pull a random sample and input RA and DEC from that. 


for i, j in zip(Ra, Dec):
  ra =i
  dec=j
  injmu_percentile = np.random.uniform(0,100)
  injduration = np.random.uniform(1,10)
  injloc_x=np.random.uniform(-5,5)
  injloc_y=np.random.uniform(-5,5)
  injpeak=10.**np.random.uniform(-3,-1)
  injmu_percentile = 20
  injduration = 2
  injloc_x = 0 #equal to negative row
  injloc_y = 4 #equal to negative col
  injpeak = 0.002

  path_to_FFIs = check_before_download(coordinates=SkyCoord(ra, dec, unit="deg"), size=50)
  path_to_FFIs = check_before_download(coordinates=SkyCoord(ra, dec, unit="deg"), size=50)
  s1 = tess_cpm.Source(path_to_FFIs[0], remove_bad=True, injection=True, injmu_percentile=injmu_percentile,
                    injduration=injduration,injloc_x=injloc_x,injloc_y=injloc_y,injpeak=injpeak)


# Need a piece of code to grid over TESS

all_time = []
all_flux = []

for delta_row in [-4,-3,-2,-1,0,1,2,3,4]:
    for delta_col in [-4, -3, -2, -1, 0, 1, 2, 3,4]:
        aux_plotting = False
        final_plotting = False
        s1.set_aperture(rowlims=[24-delta_row, 26-delta_row], collims=[24 - delta_col, 26 - delta_col])
        if aux_plotting:
            _ = s1.plot_cutout(rowlims=[20, 30], collims=[20, 30], show_aperture=True)
            _ = s1.plot_pix_by_pix()  # This method plots the raw flux values
            _ = s1.plot_pix_by_pix(data_type="normalized_flux")
        s1.add_cpm_model(exclusion_size=5, n=64, predictor_method="similar_brightness")
        s1.add_poly_model(scale=2, num_terms=4)
        s1.set_regs([0.01, 0.1])  # The first term is the CPM regularization while the second term is the polynomial regularization value.
        s1.holdout_fit_predict(k=50);  # When fitting with a polynomial component, we've found it's better to increase the number of sections.
        if aux_plotting:
            s1.plot_pix_by_pix(data_type="poly_model_prediction", split=True);
            s1.plot_pix_by_pix(data_type="cpm_subtracted_flux");
        s1_aperture_normalized_flux = s1.get_aperture_lc(data_type="normalized_flux")
        s1_aperture_cpm_prediction = s1.get_aperture_lc(data_type="cpm_prediction")
        s1_aperture_poly_prediction = s1.get_aperture_lc(data_type="poly_model_prediction")
        if aux_plotting:
            plt.plot(s1.time, s1_aperture_normalized_flux, ".", c="k", ms=8, label="Normalized Flux")
            plt.plot(s1.time, s1_aperture_cpm_prediction, "-", lw=3, c="C3", alpha=0.8, label="CPM Prediction")
            plt.plot(s1.time, s1_aperture_poly_prediction, "-", lw=3, c="C0", alpha=0.8, label="Polynomial Prediction")

            plt.xlabel("Time - 2457000 [Days]", fontsize=30)
            plt.ylabel("Normalized Flux", fontsize=30)
            plt.tick_params(labelsize=20)
            plt.legend(fontsize=30)
            plt.show()
        s1_aperture_detrended_flux = s1.get_aperture_lc(data_type="cpm_subtracted_flux")
        all_time.append(s1.time)
        all_flux.append(s1_aperture_detrended_flux)
        if final_plotting:
            plt.plot(s1.time, s1_aperture_detrended_flux, "k-")
            # plt.plot(s1.time, s1_aperture_normalized_flux-aperture_cpm_prediction, "r.", alpha=0.2)  # Gives you the same light curve as the above line
            plt.xlabel("Time - 2457000 [Days]", fontsize=30)
            plt.ylabel("CPM Flux", fontsize=30)
            plt.tick_params(labelsize=20)
            plt.tight_layout()
            plt.savefig('ra'+str(ra)+'dec'+str(dec)+'delr'+str(delta_row)+'delc'+str(delta_col)+'.png')
            plt.clf()

np.savez('test.npz',all_time=all_time,all_flux=all_flux,ra=ra,dec=dec,
            injmu_percentile=injmu_percentile,injduration=injduration,
            injloc_x=injloc_x,injloc_y=injloc_y,injpeak=injpeak)