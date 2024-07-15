from utils import FishEyeImage, rot
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, angular_separation, position_angle, offset_by, ICRS
import astropy.units as u
from PIL import Image
from scipy.optimize import minimize, curve_fit, differential_evolution

from pano import pano_fisheye

import time as nptime
file = '04'
# imgfile = '00'+str(i)+'_0333-已增强-NR'
zhangbei = EarthLocation(lon=115*u.deg+14*u.arcsec,lat=41*u.deg+13*u.arcmin+53*u.arcsec,height = 1466)
dunhuang  = EarthLocation(lon=94.322799*u.deg,lat=40.359581*u.deg,height = 1100)
lenghu = EarthLocation(lon=93.9018774741078*u.deg,lat=38.59868881470391*u.deg,height = 3500)
pic = FishEyeImage(file+'.CR3',img_path=file+'.jpg', loc = lenghu, mag_limit=6.5)


solution = pic.solve(solve_size=600)

# rms = pic.xmatch()
# pic.outlier_cliping(clip_data='a_sep', theta_range=(0,90), bin_n=10, sigma=2)
# result1  = pic.optimize()
# pic.outlier_cliping(clip_data='a_sep', theta_range=(0,70), bin_n=10, sigma=1.5)
# pic.outlier_cliping(clip_data='pa', theta_range=(40,90), bin_n=6, sigma=1.5)
# result2  = pic.distort_optimize()

# rms2 = pic.xmatch()
# pic.outlier_cliping(clip_data='a_sep', theta_range=(0,85), bin_n=7, sigma=1.5)
# pic.outlier_cliping(clip_data='a_sep', theta_range=(0,90), bin_n=15, sigma=1.5)
# result  = pic.optimize(minmize_func=differential_evolution)
# pic.draw_residual(alpha=0.3,dpi=70,s=2)

now = nptime.time()
rgb = pano_fisheye(pic,output_fov=180)
now2 = nptime.time()
print(now2-now)
# rgb2 = np.uint8(rgb*255)
# output = Image.fromarray(rgb2)
# output.save('pano'+str(i)+'.jpg')