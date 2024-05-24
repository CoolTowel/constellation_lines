from astropy.coordinates import angular_separation
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import time

hips_star = Table.read('HIP2_rad.fits')
start = time.time()

min_sep=[]
for i in range(1000):
    ra = hips_star[i]['RA']
    dec = hips_star[i]['DEC']
    others = hips_star[~(hips_star['HIP']==hips_star[i]['HIP'])]
    others = others[(others['RA']-ra)<0.1]
    others = others[(others['DEC']-dec)<0.1]
    sep = []
    for j in range(len(others)):
        sep.append(angular_separation(ra, dec,others[j]['RA'],others[j]['DEC']))
    min_sep.append(np.min(sep))

end = time.time()
total_time = end - start
print(str(total_time)+'s')

print(np.min(min_sep)/np.pi*180*60)
plt.hist(np.asarray(min_sep)/np.pi*60*180)
plt.show()