from utils import FishEyeImage
from photutils.aperture import CircularAperture
from astropy.coordinates import EarthLocation
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from PIL import Image

file = 'img-s'
img =  Image.open(file+'.jpg')
zhangbei = EarthLocation(lon=115*u.deg+14*u.arcsec,lat=41*u.deg+13*u.arcmin+53*u.arcsec,height = 1466)
dunhuang  = EarthLocation(lon=94.322799*u.deg,lat=40.359581*u.deg,height = 1100)
pic = FishEyeImage(raw_path=file+'.CR3',loc = zhangbei, mag_limit=5.5)

solution = pic.solve(solve_size=800)

print(pic.plat_para)
print(pic.lens_para)

cata_lon = pic.catalog_skycoords.az
cata_lat = pic.catalog_skycoords.alt
x,y, _,_=pic.wcs2xy(cata_lon,cata_lat)
im_u,im_v = pic.xy2uv(x,y)

plt.imshow(img)

cata = CircularAperture(np.transpose([im_u,im_v]), r=5)
cata.plot(color='red', lw=1.5)

plt.show()