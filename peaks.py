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

file = 'img-z'
img =  Image.open(file+'.jpg')
zhangbei = EarthLocation(lon=115*u.deg+14*u.arcsec,lat=41*u.deg+13*u.arcmin+53*u.arcsec,height = 1466)
dunhuang  = EarthLocation(lon=94.322799*u.deg,lat=40.359581*u.deg,height = 1100)
pic = FishEyeImage( file+'.CR3',loc = dunhuang, mag_limit=5.5)

solution = pic.solve(solve_size=1200)

sep = angular_separation(pic.az*u.rad,pic.alt*u.rad,pic.catalog_skycoords.az,pic.catalog_skycoords.alt)
sep_constarit= sep<0.45*np.pi*u.rad
selected_cata_skyc = pic.catalog_skycoords[sep_constarit]
cata_x,cata_y,_,_ = pic.az_to_delta_xy(c_az=pic.az*u.rad,c_alt=pic.alt*u.rad,az=selected_cata_skyc.az, alt=selected_cata_skyc.alt, f=pic.f, k=pic.k)

cata_xy = pic.delta_xy_to_xy(cata_x,cata_y,pic.c_x,pic.c_y,pic.az_roll)

plt.imshow(img)

cata = CircularAperture(np.transpose(cata_xy), r=5)
cata.plot(color='red', lw=1.5)

plt.show()