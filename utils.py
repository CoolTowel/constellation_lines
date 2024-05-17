import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from astropy.io import fits
from astropy.table import Table, vstack
from matplotlib.collections import LineCollection
from astropy.coordinates import angular_separation, position_angle
# from photutils.aperture import CircularAperture
from astropy.stats import sigma_clipped_stats
# from photutils.detection import find_peaks
import tetra3
import rawpy
import exiftool



t3 = tetra3.Tetra3()


def rot(vectors, deg):
    R_M = [[np.cos(deg), -np.sin(deg)],
           [np.sin(deg), np.cos(deg)]]
    return np.dot(R_M, vectors)


def tt11(theta):
    f = 14.6
    k = -0.19
    r = f*np.sin(k*theta)/k
    return r


def x_y(ra1, dec1, ra2, dec2, lens_func=tt11):
    ang_sep = angular_separation(
        ra1, dec1, ra2, dec2)
    pa = position_angle(
        ra1, dec1, ra2, dec2)
    r = lens_func(ang_sep)/0.006
    x = r * np.cos(pa)
    y = r * np.sin(pa)
    return x, y


class FishEyeImage():
    def __init__(self, img_path, raw_path, f=14.6, k=-0.19, dpi=600):
        self.k = k
        self.f = f
        with rawpy.imread(raw_path) as rawfile:
            raw = rawfile.postprocess(
                gamma=(1, 1), no_auto_bright=True, output_bps=16)
        self.img = Image.open(img_path)
        with exiftool.ExifToolHelper() as et:
            self.iso = et.get_metadata(raw_path)[0]['EXIF:ISO']

        self.raw = np.array(raw)[:, :, 1]
        self.raw[self.raw < 60000] = self.raw[self.raw <
                                              60000]//(self.iso/1600)
        # self.raw = np.array(raw)

        self.dpi = dpi

    def lens_func(self, theta):
        r = self.f*np.sin(self.k*theta)/self.k
        return r

    def solve(self, solve_size=400):
        ii = (self.raw.shape[0]-solve_size)//2
        jj = (self.raw.shape[1]-solve_size)//2

        data = self.raw[ii:ii+solve_size, jj:jj+solve_size]

        data4t3 = data.astype(np.uint16)
        data4t3[data4t3 <= 0] = 0
        img4t3 = Image.fromarray(data4t3)
        self.solution = t3.solve_from_image(img4t3, distortion=[-0.5, 0])
        print(self.solution)
        self.ra = self.solution['RA']/180*np.pi
        self.dec = self.solution['Dec']/180*np.pi
        self.roll = self.solution['Roll']/180*np.pi
        return self.solution

    def lens_proj(self, xy, image_to_show):
        xy = rot(rot(xy, np.pi/2), self.roll)
        xy = np.dot([[1, 0], [0, -1]], xy)
        xy[0] += (image_to_show.shape[1]/2)-0.5
        xy[1] += (image_to_show.shape[0]/2)-0.5
        return xy

    # def lens_proj_reverse(self, xy, image_to_show):
    #     xy

    def constellation(self, fn='test.jpg', cons_file_path='conslines.npy'):
        cons_lines = np.load(cons_file_path)
        for i in range(cons_lines.shape[0]):
            star_1 = cons_lines[i][0]
            star_2 = cons_lines[i][1]
            x1, y1 = x_y(
                self.ra, self.dec, star_1[0]/180*np.pi, star_1[1]/180*np.pi, self.lens_func)
            x2, y2 = x_y(
                self.ra, self.dec, star_2[0]/180*np.pi, star_2[1]/180*np.pi, self.lens_func)
            if x2 < x1:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            k = (y2-y1)/(x2-x1)
            disp_val = 20
            x1 += disp_val*np.cos(np.arctan(k))
            y1 += disp_val*np.sin(np.arctan(k))
            x2 -= disp_val*np.cos(np.arctan(k))
            y2 -= disp_val*np.sin(np.arctan(k))
            cons_lines[i][0] = self.lens_proj([x1, y1], self.raw)
            cons_lines[i][1] = self.lens_proj([x2, y2], self.raw)
        draw = ImageDraw.Draw(self.img)
        for i in range(cons_lines.shape[0]):
            draw.line(cons_lines[i].flatten()-16, fill='white', width=7)
        self.img.save(fn)
        # fig, ax = plt.subplots(1,1, dpi = self.dpi)
        # ax.imshow(self.img, origin='upper')

        # # ax = plt.gca()
        # ax.add_collection(
        #     LineCollection(cons_lines,
        #                    linewidths=0.75,
        #                    colors='silver'))
        # ax.axis('off')
        # fig.tight_layout()
        # plt.show()

        # plt.savefig(fn,bbox_inches='tight',pad_inches=0,dpi=self.dpi)

    # def detect_stars(self, res=500):
    #     stars_xy = Table()
    #     for i in range(self.raw.shape[0]//res):
    #         for j in range(self.raw.shape[1]//res):
    #             data = self.raw[i*res:(i+1)*res, j*res:(j+1)*res]
    #             mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    #             threshold = median + (20 * std)
    #             stars_found = find_peaks(data, threshold, box_size=11)
    #             # stars_founder = DAOStarFinder(fwhm=5, threshold=5.*std)
    #             # stars_found = stars_founder(data - median)
    #             if stars_found is not None:
    #                 stars_found['x_peak'] += j*res
    #                 stars_found['y_peak'] += i*res
    #                 # stars_found['xcentroid'] += j*res
    #                 # stars_found['ycentroid'] += i*res
    #                 stars_xy = vstack([stars_xy, stars_found])
    #     stars_eq = Table()
    #     stars_eq.add_column(stars_xy['peak_value'], name='peak_value')

    #     return stars_xy

    # def star_detect(self, res=500):
    #     self.stars = Table()
    #     for i in range(self.raw.shape[0]//res):
    #         for j in range(self.raw.shape[1]//res):
    #             data = self.raw[i*res:(i+1)*res, j*res:(j+1)*res]
    #             mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    #             threshold = median + (20 * std)
    #             stars_found = find_peaks(data, threshold, box_size=11)
    #             # stars_founder = DAOStarFinder(fwhm=5, threshold=5.*std)
    #             # stars_found = stars_founder(data - median)
    #             if stars_found is not None:
    #                 stars_found['x_peak'] += j*res
    #                 stars_found['y_peak'] += i*res
    #                 # stars_found['xcentroid'] += j*res
    #                 # stars_found['ycentroid'] += i*res

    #                 stars = vstack([stars, stars_found])

    #     self.stars.sort(keys='peak_value', reverse=True)
    #     # stars.sort(keys='mag', reverse=True)
