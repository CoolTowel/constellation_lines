from utils import FishEyeImage, rot
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, angular_separation, position_angle, offset_by, ICRS
import astropy.units as u
from PIL import Image
from utils import FishEyeImage

def equidistant(pic,theta):
    return theta*pic.lens_para['f']

def equidistant_inverse(pic, r):
    return r/pic.lens_para['f']


def pano_fisheye(pics,projection='equidistant',output_fov=200):
    match projection:
        case 'equidistant':
            proj = equidistant
            proj_inv = equidistant_inverse

    if type(pics) is list:
        pic = pics[0]
    elif type(pics) is FishEyeImage:
        pic = pics
        
    output_theta = np.pi*output_fov/180/2
    output_r = proj(pic,output_theta)
    output_size = int(output_r/pic.pixel_size)*2

    output_uv = np.mgrid[0:output_size,0:output_size]
    output_uv = np.asarray([output_uv[1],output_uv[0]])

    output_xy = output_uv-output_size//2
    output_xy.resize((2,output_size*output_size))
    output_xy = np.dot([[1, 0], [0, -1]], output_xy)
    # output_xy = rot(output_xy, - np.pi/2)
    output_x = output_xy[0]
    output_y = output_xy[1]
    r = np.sqrt(output_x**2+output_y**2)*pic.pixel_size
    pa = np.arctan2(output_y,output_x)
    theta = proj_inv(pic,r)
    alt = (pa+1.5*np.pi)*u.rad
    az = (0.5*np.pi-theta)*u.rad
    # alt, az = offset_by(0*u.deg,90*u.deg, -pa, theta)
    img = pic.img
    output_rgb = np.full((output_size,output_size,3),np.nan)

    output_in_im_x,output_in_im_y,_,_ = pic.wcs2xy(alt,az)
    output_in_im_u,output_in_im_v = pic.xy2uv(output_in_im_x,output_in_im_y)

    for i in range(3):
        im_data = np.asarray(img)[:,:,i]
        output = map_coordinates(input=im_data,coordinates=[output_in_im_v,output_in_im_u],cval=np.nan) 
        # coordinates = [output_in_im_v, output_in_im_u] 注意这里u和v是反的
        output_rgb[:,:,i]  = output.reshape((output_size,output_size))/255

    return output_rgb
