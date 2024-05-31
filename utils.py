import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from astropy.io import fits
from astropy.table import Table, vstack
from matplotlib.collections import LineCollection
from astropy.coordinates import angular_separation, position_angle, offset_by, SkyCoord
# from photutils.aperture import CircularAperture
from astropy.stats import sigma_clipped_stats
# from photutils.detection import find_peaks
import tetra3
import rawpy
import exiftool
from photutils.detection import find_peaks, DAOStarFinder
import astropy.units as u
from scipy.optimize import minimize

t3 = tetra3.Tetra3()


def rot(vectors, deg):
    R_M = [[np.cos(deg), -np.sin(deg)],
           [np.sin(deg), np.cos(deg)]]
    return np.dot(R_M, vectors)


def x_y(ra1, dec1, ra2, dec2, lens_func, pixel_size=0.006):
    ang_sep = angular_separation(
        ra1, dec1, ra2, dec2)
    pa = position_angle(
        ra1, dec1, ra2, dec2)
    r = lens_func(ang_sep)/pixel_size
    x = r * np.cos(pa)
    y = r * np.sin(pa)
    return x, y, ang_sep


class FishEyeImage():
    def __init__(self, img_path, raw_path, raw_iso_corr=False, f=14.6, k=-0.19, pixel_size=0.006, sensor='full_frame', star_catalog='HIP2_rad.fits', mag_limit=6.5):
        self.k = k
        self.f = f
        with rawpy.imread(raw_path) as rawfile:
            raw = rawfile.postprocess(
                gamma=(1, 1), no_auto_bright=True, output_bps=16)[16:4016, 20:6020]
        self.img = Image.open(img_path)
        with exiftool.ExifToolHelper() as et:
            self.iso = et.get_metadata(raw_path)[0]['EXIF:ISO']

        self.raw = np.mean(np.asarray(raw), axis=-1)
        if raw_iso_corr:
            self.raw[self.raw < 60000] = self.raw[self.raw <
                                                  60000]//(self.iso/1600)
        # self.raw = np.array(raw)
        if pixel_size is not None:
            self.pixel_size = pixel_size
        else:
            if sensor == 'full_frame':
                self.pixel_size = 24/self.raw.shape[0]
            elif sensor == 'apsc':
                self.pixel_size = 24/self.raw.shape[0]/1.55

        self.catalog = Table.read(star_catalog)
        self.catalog = self.catalog[np.logical_and(
            self.catalog['Hpmag'] < mag_limit, self.catalog['Hpmag'] > 1)]

    def lens_func(self, theta, f=None, k=None):  # see https://ptgui.com/support.html#3_28
        if k == None or f == None:
            if self.k >= -1 and self.k < 0:
                r = self.f*np.sin(self.k*theta)/self.k
            elif self.k == 0:
                r = self.f*theta
            elif self.k > 0 and self.k <= 1:
                r = self.f*np.tan(self.k*theta)/self.k
        else:
            if k >= -1 and k < 0:
                r = f*np.sin(k*theta)/k
            elif k == 0:
                r = f*theta
            elif k > 0 and k <= 1:
                r = f*np.tan(k*theta)/k
        return r

    def reverse_lens_func(self, r, f=None, k=None):
        if k == None or f == None:
            if self.k >= -1 and self.k < 0:
                theta = np.arcsin(self.k/self.f*r)/self.k
        else:
            if k >= -1 and k < 0:
                theta = np.arcsin(k/f*r)/k
        return theta

    def solve(self, solve_size=400):
        ii = (self.raw.shape[0]-solve_size)//2
        jj = (self.raw.shape[1]-solve_size)//2

        data = self.raw[ii:ii+solve_size, jj:jj+solve_size]

        data4t3 = data.astype(np.uint16)
        data4t3[data4t3 <= 0] = 0
        img4t3 = Image.fromarray(data4t3)
        self.solution = t3.solve_from_image(img4t3, distortion=-0.0085542)
        print(self.solution)
        self.ra = self.solution['RA']/180*np.pi
        self.dec = self.solution['Dec']/180*np.pi
        self.roll = self.solution['Roll']/180*np.pi
        return self.solution

    def rot_shift(self, xy, image_to_show):
        xy = rot(rot(xy, np.pi/2), self.roll)
        xy = np.dot([[1, 0], [0, -1]], xy)
        xy[0] += (image_to_show.shape[1]/2)
        xy[1] += (image_to_show.shape[0]/2)
        return xy

    def detect_stars(self, res=500):
        self.stars_xy = Table()
        for i in range(self.raw.shape[0]//res):
            for j in range(self.raw.shape[1]//res):
                data = self.raw[i*res:(i+1)*res, j*res:(j+1)*res]
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                threshold = median + (20 * std)
                # stars_found = find_peaks(data, threshold, box_size=11)
                stars_founder = DAOStarFinder(fwhm=5, threshold=threshold)
                stars_found = stars_founder(data)
                if stars_found is not None:
                    # stars_found['x_peak'] += j*res
                    # stars_found['y_peak'] += i*res
                    stars_found['xcentroid'] += j*res
                    stars_found['ycentroid'] += i*res
                    self.stars_xy = vstack([self.stars_xy, stars_found])
        # stars_eq.add_column(stars_xy['peak_value'], name='peak_value')
        return self.stars_xy

    def detect_stars_eq(self):
        stars_x = self.stars_xy['xcentroid']
        stars_y = self.stars_xy['ycentroid']
        delta_x = stars_x-self.raw.shape[1]/2
        delta_y = stars_y-self.raw.shape[0]/2
        delta_xy = np.asarray([delta_x, delta_y])
        delta_xy = np.dot([[1, 0], [0, -1]], delta_xy)
        delta_xy = rot(delta_xy, -self.roll-np.pi/2)
        r = np.sqrt(delta_x**2+delta_y**2)*self.pixel_size
        pa = np.arctan2(delta_xy[1], delta_xy[0])
        pa[pa < 0] += 2*np.pi
        theta = self.reverse_lens_func(r)
        ra, dec = offset_by(self.ra, self.dec, pa, theta)
        self.stars_eq = Table()
        self.stars_eq.add_columns([ra, dec], names=['RAdeg', 'DECdeg'])

    def first_match(self, mag_limit=6.5):
        detect_star_skycoords = SkyCoord(
            ra=self.stars_eq['RAdeg'], dec=self.stars_eq['DECdeg'], frame='icrs')
        catalog_skycoords = SkyCoord(
            ra=self.catalog['RA'], dec=self.catalog['DEC'], frame='icrs', unit='rad')
        idx, d2d, d3d = catalog_skycoords.match_to_catalog_sky(
            detect_star_skycoords)
        return idx, d2d, d3d

    def plate_optimize(self, ra_dec_range=2, roll_range=3,f_range=0.1,k_range=0.1):
        idx, d2d, d3d = self.first_match()
        stars_to_be_use = self.catalog[np.where(d2d.to(u.arcmin) < 20*u.arcmin)[0]]
        catalog_skycoords = SkyCoord(
            ra=stars_to_be_use['RA'], dec=stars_to_be_use['DEC'], frame='icrs', unit='rad')
        ra_dec_range = ra_dec_range/180*np.pi
        roll_range = roll_range/180*np.pi

        def star_rms(parameters):
            """
            x = [ra,dec,roll,f,k]
            """
            stars_x = self.stars_xy['xcentroid']
            stars_y = self.stars_xy['ycentroid']
            delta_x = stars_x-self.raw.shape[1]/2
            delta_y = stars_y-self.raw.shape[0]/2
            delta_xy = np.asarray([delta_x, delta_y])
            delta_xy = np.dot([[1, 0], [0, -1]], delta_xy)
            delta_xy = rot(delta_xy, -parameters[2]-np.pi/2)
            r = np.sqrt(delta_x**2+delta_y**2)*self.pixel_size
            pa = np.arctan2(delta_xy[1], delta_xy[0])
            pa[pa < 0] += 2*np.pi
            theta = self.reverse_lens_func(r, f=parameters[3], k=parameters[4])
            stars_ra, stars_dec = offset_by(parameters[0], parameters[1], pa, theta)
            stars_skycoords = SkyCoord(
                ra=stars_ra, dec=stars_dec, frame='icrs')
            idx, d2d, d3d = catalog_skycoords.match_to_catalog_sky(
                stars_skycoords)
            d2d = d2d.to(u.arcsec).value
            return np.sqrt(np.mean(d2d**2))
        result = minimize(fun=star_rms, x0=np.asarray([self.ra, self.dec, self.roll, self.f, self.k]), bounds=(
            (self.ra-ra_dec_range/2, self.ra+ra_dec_range/2), 
            (self.dec-ra_dec_range/2, self.dec+ra_dec_range/2), 
            (self.roll-roll_range/2, self.roll+roll_range/2), 
            (self.f-f_range/2, self.f+f_range/2), 
            (self.k-k_range/2, self.k+k_range/2))
            )

        return [self.ra, self.dec, self.roll, self.f, self.k], result

    def constellation(self, fn='test.jpg', cons_file_path='conslines.npy'):
        cons_lines = np.load(cons_file_path)
        cons_lines_xy = np.array([[0, 0], [0, 0]])
        draw = ImageDraw.Draw(self.img)

        for i in range(cons_lines.shape[0]):
            star_1 = cons_lines[i][0]
            star_2 = cons_lines[i][1]
            x1, y1, angular_separation1 = x_y(
                self.ra, self.dec, star_1[0]/180*np.pi, star_1[1]/180*np.pi, self.lens_func, self.pixel_size)
            x2, y2, angular_separation2 = x_y(
                self.ra, self.dec, star_2[0]/180*np.pi, star_2[1]/180*np.pi, self.lens_func, self.pixel_size)
            if angular_separation1 < 0.45*np.pi or angular_separation2 < 0.45*np.pi:
                if x2 < x1:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                k = (y2-y1)/(x2-x1)
                break_for_star = 20  # 星座连线断开，露出恒星
                x1 += break_for_star*np.cos(np.arctan(k))
                y1 += break_for_star*np.sin(np.arctan(k))
                x2 -= break_for_star*np.cos(np.arctan(k))
                y2 -= break_for_star*np.sin(np.arctan(k))
                x1, y1 = self.rot_shift([x1, y1], self.raw)
                x2, y2 = self.rot_shift([x2, y2], self.raw)
                # cons_lines_xy = np.append(cons_lines_xy,[line_vertex1,line_vertex2])
                # cons_lines[i][1] = self.lens_proj([x2, y2], self.raw)
                draw.line([x1, y1, x2, y2], fill='white', width=7)
        self.img.save(fn)
