from astroquery.mast import Tesscut
from astroquery.mast.utils import parse_input_location
import os
from astropy.coordinates import SkyCoord
import tess_cpm
import matplotlib.pyplot as plt


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
ra = 64.525833
dec = -63.615669

#Need to code to pull down the correct sectors
path_to_FFIs = check_before_download(coordinates=SkyCoord(ra, dec, unit="deg"), sector=1, size=50)
path_to_FFIs = check_before_download(coordinates=SkyCoord(ra, dec, unit="deg"), sector=2, size=50)
s1 = tess_cpm.Source(path_to_FFIs[0], remove_bad=True)
#_ = s1.plot_cutout()


# Need a piece of code to grid over TESS
delta_row = 0
delta_col = -4
s1.set_aperture(rowlims=[24-delta_row, 26-delta_row], collims=[24 - delta_col, 26 - delta_col])
_ = s1.plot_cutout(rowlims=[20, 30], collims=[20, 30], show_aperture=True)
_ = s1.plot_pix_by_pix()  # This method plots the raw flux values
_ = s1.plot_pix_by_pix(data_type="normalized_flux")
s1.add_cpm_model(exclusion_size=5, n=64, predictor_method="similar_brightness")
s1.add_poly_model(scale=2, num_terms=4)
s1.set_regs([0.01, 0.1])  # The first term is the CPM regularization while the second term is the polynomial regularization value.
s1.holdout_fit_predict(k=50);  # When fitting with a polynomial component, we've found it's better to increase the number of sections.
s1.plot_pix_by_pix(data_type="poly_model_prediction", split=True);
s1.plot_pix_by_pix(data_type="cpm_subtracted_flux");
s1_aperture_normalized_flux = s1.get_aperture_lc(data_type="normalized_flux")
s1_aperture_cpm_prediction = s1.get_aperture_lc(data_type="cpm_prediction")
s1_aperture_poly_prediction = s1.get_aperture_lc(data_type="poly_model_prediction")
plt.plot(s1.time, s1_aperture_normalized_flux, ".", c="k", ms=8, label="Normalized Flux")
plt.plot(s1.time, s1_aperture_cpm_prediction, "-", lw=3, c="C3", alpha=0.8, label="CPM Prediction")
plt.plot(s1.time, s1_aperture_poly_prediction, "-", lw=3, c="C0", alpha=0.8, label="Polynomial Prediction")

plt.xlabel("Time - 2457000 [Days]", fontsize=30)
plt.ylabel("Normalized Flux", fontsize=30)
plt.tick_params(labelsize=20)
plt.legend(fontsize=30)
plt.show()
s1_aperture_detrended_flux = s1.get_aperture_lc(data_type="cpm_subtracted_flux")
plt.plot(s1.time, s1_aperture_detrended_flux, "k-")
# plt.plot(s1.time, s1_aperture_normalized_flux-aperture_cpm_prediction, "r.", alpha=0.2)  # Gives you the same light curve as the above line
plt.xlabel("Time - 2457000 [Days]", fontsize=30)
plt.ylabel("CPM Flux", fontsize=30)
plt.tick_params(labelsize=20)
plt.show()