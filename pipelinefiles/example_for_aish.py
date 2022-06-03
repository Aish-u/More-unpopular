import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_training_data(load=False, datafile ='test.npz'):
    
    if not load:
        # This grabs the training data files from the appropriate directory
        mypath = './training_data/'
        training_data_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        
        training_t = []
        training_f = []
        training_class = []
        training_info = []
        for f in training_data_files:
            data = np.load(mypath+f)
            all_time=data['all_time']
            all_flux=data['all_flux']
            ra = data['ra']
            dec = data['dec']
            injmu_percentile=data['injmu_percentile']
            injduration=data['injduration']
            injloc_x=-1 * data['injloc_x']
            injloc_y=-1 * data['injloc_y']
            injpeak=data['injpeak']
            
            info = ra,dec,injmu_percentile,injduration,injloc_x,injloc_y,injpeak
            for i, t in enumerate(all_time):
                f = all_flux[i]
                row = np.floor(i/9) - 4
                col = i%9 - 4
                if np.all(np.isnan(f)):
                    continue
                if np.sqrt((row - injloc_x)**2 + (col-injloc_y)**2)<=2:
                    training_t.append(t)
                    training_f.append(f)
                    training_class.append(1)
                    training_info.append(info)
                    
                elif np.sqrt((row - injloc_x)**2 + (col-injloc_y)**2)>5:
                    training_t.append(t)
                    training_f.append(f)
                    training_class.append(0)
                    training_info.append(info)
        np.savez(datafile,training_t = training_t, training_f = training_f,
                training_class = training_class, training_info = training_info)
        return training_t, training_f,training_class,training_info
    else:
        data = np.load(datafile, allow_pickle=True)
        training_t = data['training_t']
        training_f = data['training_f']
        training_class = data['training_class']
        training_info = data['training_info']
        return training_t, training_f,training_class,training_info
    
training_t, training_f,training_class,training_info = get_training_data(load=True)

all_my_guesses = []
for i in np.arange(len(training_t)):
    if np.nanmean(training_f[i])> 1.0:
        my_guess = 1 #I think there is a transient!
    else:
        my_guess = 0 #I think there is NOT a transient
    all_my_guesses.append(my_guess)
#" training class" are the true classes, where 1 means there IS a transient, and 0 means NO transient
cm = confusion_matrix(training_class, all_my_guesses)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('cm.png')