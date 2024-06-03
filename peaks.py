from utils_az import FishEyeImage, rot
from photutils.detection import find_peaks, DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.coordinates import angular_separation, position_angle, EarthLocation
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from PIL import Image, ImageDraw

hips_star = Table.read('HIP2_rad.fits')

file = '291A2094'
img =  Image.open(file+'.jpg')
zhangbei = EarthLocation(lon=115*u.deg+14*u.arcsec,lat=41*u.deg+13*u.arcmin+53*u.arcsec,height = 1466)
dunhuang  = EarthLocation(lon=94.322799*u.deg,lat=40.359581*u.deg,height = 1100)
pic = FishEyeImage( file+'.CR3',loc = zhangbei,mag_limit=3.5)

solution = pic.solve(solve_size=1200)

# star_xy, star_az = pic.detect_stars_az()

# pic.first_match()
sep = angular_separation(pic.az*u.rad,pic.alt*u.rad,pic.catalog_skycoords.az,pic.catalog_skycoords.alt)
sep_constarit= sep<0.45*np.pi*u.rad
selected_cata_skyc = pic.catalog_skycoords[sep_constarit]
cata_x,cata_y,_,_ = pic.az_to_delta_xy(c_az=pic.az*u.rad,c_alt=pic.alt*u.rad,az=selected_cata_skyc.az, alt=selected_cata_skyc.alt, f=pic.f, k=pic.k)

cata_xy = pic.delta_xy_to_xy(cata_x,cata_y,pic.c_x,pic.c_y,pic.az_roll)

plt.imshow(img)

# stars = CircularAperture(np.transpose(star_xy), r=2)
# stars.plot(color='blue', lw=1.5)

cata = CircularAperture(np.transpose(cata_xy), r=3)
cata.plot(color='red', lw=1.5)

plt.show()
# pic.detect_stars_az()
# init , final = pic.plate_optimize(ra_dec_range=3, roll_range=4,f_range=50,k_range=4)
# final = final.x
# ra = final[0]
# dec= final[1]

# def hip_stars(ra,dec, max_mag, min_mag):
#     hips_mag = hips_star[np.logical_and(
#         hips_star['Hpmag'] <= min_mag, hips_star['Hpmag'] >= max_mag)]
#     ang_sep_list = []
#     for i in range(len(hips_mag)):
#         ang_sep = angular_separation(
#             ra, dec, hips_mag[i]['RA'], hips_mag[i]['DEC'])
#         ang_sep_list.append(ang_sep)
#     hips_mag.add_column(ang_sep_list, name='seprataion')
#     stars_table = hips_mag[hips_mag['seprataion']
#                            < 0.45*np.pi]

#     pa_list = []
#     for i in range(len(stars_table)):
#         pa = position_angle(
#             ra, dec, stars_table[i]['RA'], stars_table[i]['DEC'])
#         pa_list.append(pa)
#     stars_table.add_column(pa_list, name='position_angle')

#     return stars_table

# def detect_stars(self, res=500):
#     stars_xy = Table()
#     for i in range(self.raw.shape[0]//res):
#         for j in range(self.raw.shape[1]//res):
#             data = self.raw[i*res:(i+1)*res, j*res:(j+1)*res]
#             mean, median, std = sigma_clipped_stats(data, sigma=3.0)
#             threshold = median + (15 * std)
#             # stars_found = find_peaks(data, threshold, box_size=11)
#             stars_founder = DAOStarFinder(fwhm=5, threshold=20.*std)
#             stars_found = stars_founder(data - median)
#             if stars_found is not None:
#                 # stars_found['x_peak'] += j*res
#                 # stars_found['y_peak'] += i*res
#                 stars_found['xcentroid'] += j*res
#                 stars_found['ycentroid'] += i*res
#                 stars_xy = vstack([stars_xy, stars_found])
#     stars_eq = Table()
#     # stars_eq.add_column(stars_xy['peak_value'], name='peak_value')
#     return stars_xy
# stars_det = pic.detect_stars()
# stars_det_bright = stars_det
# stars_det_positions = np.transpose(
#     (stars_det_bright['xcentroid'], stars_det_bright['ycentroid']))


# val_stars = hip_stars(ra,dec,max_mag=-1,min_mag=6.5)
# r = pic.lens_func(val_stars['seprataion'],f=final[3],k=final[4])/0.006
# x = r * np.cos(val_stars['position_angle'])
# y = r * np.sin(val_stars['position_angle'])

# roll = final[2]


# plt.show()