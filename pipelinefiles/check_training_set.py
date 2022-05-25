import numpy as np
import matplotlib.pyplot as plt

data = np.load('test.npz')

all_time=data['all_time']
all_flux=data['all_flux']
ra = data['ra']
dec = data['dec']
injmu_percentile=data['injmu_percentile']
injduration=data['injduration']
injloc_x=data['injloc_x']
injloc_y=data['injloc_y']
injpeak=data['injpeak']

print(ra,dec,injmu_percentile,injduration,injloc_x,injloc_y,injpeak)
for i, t in enumerate(all_time):
    fig, ax = plt.subplots()
    f = all_flux[i]
    plt.plot(t,f)
    row = np.floor(i/9) - 4
    col = i%9 - 4
    plt.title(str(row)+' '+str(col))
    ax.axvspan(np.percentile(t,injmu_percentile) - \
                injduration * 2.0, np.percentile(t,injmu_percentile) + \
                injduration * 2.0, alpha=0.5, color='green')
    plt.savefig('lc'+str(row)+'_'+str(col)+'.png')
    plt.clf()
    plt.close()