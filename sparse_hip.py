from astropy.coordinates import angular_separation
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

hips_star = Table.read('HIP2_rad.fits')
start = time.time()

hips_star = hips_star[hips_star['Hpmag'] < 6]

ra_all = hips_star['RA']
dec_all = hips_star['DEC']
hip_all = hips_star['HIP']


def calculate_min_separation(i):
    ra = ra_all[i]
    dec = dec_all[i]
    hip = hip_all[i]
    delta_ra = np.abs(ra_all - ra)
    mask = delta_ra > np.pi
    delta_ra[mask] = 2*np.pi - delta_ra[mask]
    mask = (hip_all != hip) & (delta_ra < 0.2) & (np.abs(dec_all - dec) < 0.2)
    if np.sum(mask) == 0:
        return 0.3

    others_ra = ra_all[mask]
    others_dec = dec_all[mask]

    sep = angular_separation(ra, dec, others_ra, others_dec)
    return np.min(sep)


min_sep = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(calculate_min_separation, i)
               for i in range(len(hips_star))]
    for future in futures:
        min_sep.append(future.result())


hips_star.add_column(min_sep, name='min_sep')
hips_star.add_column(np.asarray(min_sep)/np.pi*180*60, name='min_sep_arcmin')
hips_star.write('hip2_6mag_minsep.fits', format='fits',overwrite=True)

hips_star_sparse = hips_star[hips_star['min_sep_arcmin']>160]
hips_star_sparse.write('hip2_6mag_sparse.fits', format='fits',overwrite=True)
np.save('min_sep.npy', min_sep)
end = time.time()
total_time = end - start
print(f"Total time: {total_time:.2f}s")
