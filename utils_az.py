import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from astropy.io import fits
from astropy.table import Table, vstack
from matplotlib.collections import LineCollection
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, angular_separation, position_angle, offset_by, ICRS
# from photutils.aperture import CircularAperture
from astropy.stats import sigma_clipped_stats
# from photutils.detection import find_peaks
import tetra3
import rawpy
import exiftool
from photutils.detection import find_peaks, DAOStarFinder
import astropy.units as u
from astropy.time import Time

from scipy.optimize import minimize, curve_fit, fsolve

t3 = tetra3.Tetra3()


def alt2pres(altitude):
    # https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/atmosphere.html
    press = 100 * ((44331.514 - altitude) / 11880.516) ** (1 / 0.1902632)

    return press/100*u.hPa


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
    def __init__(self, raw_path, loc, cutout=False, raw_iso_corr=False, f=14.6, k=-0.19, pixel_size=0.006, sensor='full_frame', star_catalog='HIP2_rad.fits', mag_limit=6.5):
        self.k = k
        self.f = f
        self.loc = loc
        with rawpy.imread(raw_path) as rawfile:
            raw = rawfile.postprocess(
                gamma=(1, 1), no_auto_bright=True, output_bps=16)[16:4016, 20:6020]

        with exiftool.ExifToolHelper() as et:
            exif = et.get_metadata(raw_path)[0]
            time = exif['EXIF:DateTimeOriginal']
            offset = exif['EXIF:OffsetTime']
        time = time.replace(':', '-', 2)
        self.obstime = Time(time)-int(offset[0:3])*u.hour
        self.raw = np.mean(np.asarray(raw), axis=-1)

        self.height = self.raw.shape[0]
        self.width = self.raw.shape[1]
        if cutout:
            ii = (self.height-cutout)//2
            jj = (self.width-cutout)//2
            self.raw = self.raw[ii:ii+cutout, jj:jj+cutout]
            self.height = self.raw.shape[0]
            self.width = self.raw.shape[1]

        if raw_iso_corr:
            self.raw[self.raw < 60000] = self.raw[self.raw <
                                                  60000]//(self.iso/1600)

        self.c_x = self.raw.shape[1]/2
        self.c_y = self.raw.shape[0]/2
        if pixel_size is not None:
            self.pixel_size = pixel_size
        else:
            if sensor == 'full_frame':
                self.pixel_size = 24/self.height
            elif sensor == 'apsc':
                self.pixel_size = 24/self.height/1.55

        self.catalog = Table.read(star_catalog)
        self.catalog = self.catalog[np.logical_and(
            self.catalog['Hpmag'] < mag_limit, self.catalog['Hpmag'] > 1)]
        self.catalog_skycoords = SkyCoord(
            ra=self.catalog['RA'], dec=self.catalog['DEC'], frame='icrs', unit='rad')
        self.az_frame = AltAz(obstime=self.obstime, location=self.loc, pressure=alt2pres(
            self.loc.height.value), temperature=0*u.deg_C, relative_humidity=0.5, obswl=550*u.nm)
        self.catalog_skycoords = self.catalog_skycoords.transform_to(
            self.az_frame)
        self.ks = [0]*5

    def solve(self, solve_size=800):
        ii = (self.height-solve_size)//2
        jj = (self.width-solve_size)//2
        data = self.raw[ii:ii+solve_size, jj:jj+solve_size]
        data4t3 = data.astype(np.uint16)
        data4t3[data4t3 <= 0] = 0
        img4t3 = Image.fromarray(data4t3)
        self.solution = t3.solve_from_image(img4t3, distortion=-0.008)
        print(self.solution)
        self.center_ra = self.solution['RA']/180*np.pi
        self.center_dec = self.solution['Dec']/180*np.pi
        self.eq_roll = self.solution['Roll']/180*np.pi
        center_eq = SkyCoord(ra=self.center_ra,
                             dec=self.center_dec, frame='icrs', unit='rad')
        self.center_az = center_eq.transform_to(self.az_frame)
        az_of_zenith = SkyCoord(az=0*u.deg, alt=90*u.deg, frame=self.az_frame)
        eq_of_zenith = az_of_zenith.transform_to(ICRS)

        eq_of_north = SkyCoord(ra=0, dec=90, frame='icrs', unit='deg')
        az_of_north = eq_of_north.transform_to(self.az_frame)

        cos_c = np.cos(angular_separation(az_of_north.az,az_of_north.alt,az_of_zenith.az,az_of_zenith.alt))
        a = angular_separation(self.center_az.az,self.center_az.alt,az_of_north.az,az_of_north.alt)
        b = angular_separation(self.center_az.az,self.center_az.alt,az_of_zenith.az,az_of_zenith.alt)
        cos_a = np.cos(a)
        cos_b = np.cos(b)
        sin_a = np.sin(a)
        sin_b = np.sin(b)
        cos_C = (cos_c-cos_a*cos_b)/(sin_a*sin_b)
        C = np.arccos(cos_C).value
        # if C<0:
        #     C += 2*np.pi
        self.az_roll = self.eq_roll-C
        # https://en.wikipedia.org/wiki/Spherical_law_of_cosines
        self.az = self.center_az.az.to('rad').value
        self.alt = self.center_az.alt.to('rad').value

        return self.solution

    # see https://ptgui.com/support.html#3_28
    def lens_func(self, theta, f, k, ks=[0]*5):
        # r = f*np.sin(k*theta)/k
        def func(r,theta):
            return np.arcsin(r*k/f)/k-(ks[0]*r**5 + ks[1]*r**4 + ks[2]*r**3 + ks[3]*r**2 + ks[4]*r)-theta
        r = []
        for y in theta:
            r.append(fsolve(func, 10,args=(y))[0])
        # return r
        return np.asarray(r)

    def reverse_lens_func(self, r, f, k, ks):
        theta = np.arcsin(r*k/f)/k
        theta -= (ks[0]*r**5 + ks[1]*r**4 + ks[2]*r**3 + ks[3]*r**2 + ks[4]*r)
        return theta

    def rot_shift(self, xy, image_to_show):
        xy = rot(rot(xy, np.pi/2), self.roll)
        xy = np.dot([[1, 0], [0, -1]], xy)
        xy[0] += (image_to_show.shape[1]/2)
        xy[1] += (image_to_show.shape[0]/2)
        return xy

    def az_to_delta_xy(self, c_az, c_alt, az, alt, f, k, ks=[0]*5):
        ang_sep = angular_separation(c_az, c_alt, az, alt)
        pa = position_angle(c_az, c_alt, az, alt)
        r = self.lens_func(ang_sep.to(u.rad).value, f, k, ks)/self.pixel_size
        x = r * np.cos(pa)
        y = r * np.sin(pa)
        return x, y, ang_sep.to(u.arcmin), pa.to(u.arcmin)

    def xy_to_delta_xy(self, x, y, c_x, c_y, roll):
        delta_x = x - c_x
        delta_y = y - c_y
        delta_xy = np.asarray([delta_x, delta_y])
        delta_xy = np.dot([[1, 0], [0, -1]], delta_xy)
        delta_xy = rot(delta_xy, -roll-np.pi/2)
        return delta_xy
    
    def delta_xy_to_xy(self, delta_x, delta_y, c_x, c_y, roll):
        delta_xy = np.asarray([delta_x, delta_y])
        delta_xy = rot(delta_xy, roll+np.pi/2)
        delta_xy = np.dot([[1, 0], [0, -1]], delta_xy)
        x = delta_xy[0] + c_x
        y = delta_xy[1] + c_y
        return [x,y]

    def xy_to_az(self, c_az, c_alt, x, y, c_x, c_y, roll, f, k, ks=[0]*5):
        delta_xy = self.xy_to_delta_xy(x, y, c_x, c_y, roll)
        r = np.sqrt(delta_xy[0]**2+delta_xy[1]**2)*self.pixel_size
        pa = np.arctan2(delta_xy[1], delta_xy[0])
        pa[pa < 0] += 2*np.pi
        theta = self.reverse_lens_func(r, f, k, ks)
        az, alt = offset_by(c_az, c_alt, pa, theta)
        return az, alt, (theta*u.rad).to(u.arcmin), (pa*u.rad).to(u.arcmin)

    def detect_stars(self, res=500):
        self.stars_xy = Table()
        for i in range(self.height//res):
            for j in range(self.width//res):
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

    def detect_stars_az(self):
        self.detect_stars()
        az, alt, _, _ = self.xy_to_az(
            c_az=self.az, c_alt=self.alt, x=self.stars_xy['xcentroid'], y=self.stars_xy['ycentroid'],
            c_x=self.c_x, c_y=self.c_y, roll=self.az_roll, f=self.f, k=self.k)
        self.detect_star_skycoords = SkyCoord(
            az=az, alt=alt, frame=self.az_frame)
        return [self.stars_xy['xcentroid'],self.stars_xy['ycentroid']], self.detect_star_skycoords

    def first_match(self, bin_size=75, max_sep=20):
        self.detect_stars_az()
        idx, d2d, _ = self.catalog_skycoords.match_to_catalog_sky(
            self.detect_star_skycoords)
        max_sep = max_sep*u.arcmin
        sep_constraint = d2d < max_sep
        self.matched_catalog_skycoords = self.catalog_skycoords[sep_constraint]
        # self.matched_detect_stars_skycoords = self.detect_star_skycoords[idx[sep_constraint]]
        self.matched_stars_xy = self.stars_xy[idx[sep_constraint]]
        catalog_theta = angular_separation(
            self.az*u.rad, self.alt*u.rad, self.matched_catalog_skycoords.az, self.matched_catalog_skycoords.alt).to(u.deg)
        uniform_ids = np.asarray([])
        for i in range(6):
            theta_min = i*15*u.deg
            theta_max = (1+i)*15*u.deg
            ids = np.where((catalog_theta >= theta_min) &
                           (catalog_theta < theta_max))[0]
            if len(ids) > bin_size:
                uniform_ids = np.append(
                    uniform_ids, np.random.choice(ids, 75, replace=False))
            else:
                uniform_ids = np.append(uniform_ids, ids)
        uniform_ids = uniform_ids.astype(int)
        self.matched_catalog_skycoords = self.matched_catalog_skycoords[uniform_ids]
        self.matched_stars_xy = self.matched_stars_xy[uniform_ids]
        return idx, d2d

    def draw_residual(self, dpi, **kwargs):
        stars_az, stars_alt, stars_theta, stars_pa = self.xy_to_az(
            c_az=self.az, c_alt=self.alt,
            x=self.matched_stars_xy['xcentroid'], y=self.matched_stars_xy['ycentroid'],
            c_x=self.c_x, c_y=self.c_y,
            roll=self.az_roll, f=self.f, k=self.k
        )
        ang_sep = angular_separation(
            stars_az, stars_alt, self.matched_catalog_skycoords.az, self.matched_catalog_skycoords.alt)

        catalog_delta_x, catalog_delta_y, catalog_theta, catalog_pa = self.az_to_delta_xy(
            c_az = self.az*u.rad, 
            c_alt = self.alt*u.rad,
            az = self.matched_catalog_skycoords.az, 
            alt = self.matched_catalog_skycoords.alt,
            f = self.f, k = self.k
        )

        stars_delta_xy = self.xy_to_delta_xy(
            self.matched_stars_xy['xcentroid'], self.matched_stars_xy['ycentroid'], self.c_x, self.c_y, self.az_roll)
        xy_sep = np.sqrt(
            (stars_delta_xy[0]-catalog_delta_x)**2+(stars_delta_xy[1]-catalog_delta_y)**2)

        theta_rs = stars_theta-catalog_theta
        pa_rs = stars_pa-catalog_pa

        # catalog_theta = angular_separation(
        #     parameters[0]*u.rad, parameters[1]*u.rad, self.matched_catalog_stars['RA'], self.matched_catalog_stars['DEC']).to(u.deg)
        fig, axs = plt.subplots(2, 2, dpi=dpi)
        for ax in axs.flatten():
            ax.set_xlabel('theta to center [deg]')
            ax.set_xlim(0, 90)

        axs[0, 0].scatter(catalog_theta.to(u.deg),
                          ang_sep.to(u.arcmin), **kwargs)
        axs[0, 0].set_ylabel('angular sepration [arcmin]')
        axs[0, 1].scatter(catalog_theta.to(u.deg), xy_sep, **kwargs)
        axs[0, 1].set_ylabel('xy sepration [pixel]')
        axs[1, 0].scatter(catalog_theta.to(u.deg), theta_rs, **kwargs)
        axs[1, 0].set_ylabel('theta res [arcmin]')
        axs[1, 1].scatter(catalog_theta.to(u.deg), pa_rs, **kwargs)
        axs[1, 1].set_ylabel('position angle res [arcmin]')
        axs[0, 0].set_ylim(0, 30)
        axs[0, 1].set_ylim(0, 20)
        axs[1, 0].set_ylim(-25, 25)
        axs[1, 1].set_ylim(-30, 30)

        plt.tight_layout()
        plt.show()

    # def draw_residual_dis(self, dpi, **kwargs):
    #     stars_ra, stars_dec, matched_stars_theta, matched_stars_pa = self.xy_to_eq_distort(
    #         self.matched_stars_xy['xcentroid'], self.matched_stars_xy['ycentroid'])
    #     stars_skycoords = SkyCoord(
    #         ra=stars_ra, dec=stars_dec, frame='icrs')
    #     _, ang_sep, _ = self.matched_catalog_skycoords.match_to_catalog_sky(
    #         stars_skycoords)

    #     matched_stars_delta_x = self.matched_stars_xy['xcentroid'] - self.c_x
    #     matched_stars_delta_y = self.matched_stars_xy['ycentroid'] - self.c_y
    #     matched_stars_delta_xy = np.asarray(
    #         [matched_stars_delta_x, matched_stars_delta_y])
    #     matched_stars_delta_xy = np.dot(
    #         [[1, 0], [0, -1]], matched_stars_delta_xy)
    #     matched_stars_delta_xy = rot(
    #         matched_stars_delta_xy, -self.roll-np.pi/2)
    #     catalog_x, catalog_y, catalog_theta, catalog_pa = self.eq_to_xy(
    #         self.ra*u.rad, self.dec*u.rad, self.matched_catalog_skycoords.ra, self.matched_catalog_skycoords.dec)
    #     xy_sep = np.sqrt(
    #         (matched_stars_delta_xy[0]-catalog_x)**2+(matched_stars_delta_xy[1]-catalog_y)**2)

    #     theta_rs = matched_stars_theta-catalog_theta
    #     pa_rs = matched_stars_pa-catalog_pa

    #     # catalog_theta = angular_separation(
    #     #     parameters[0]*u.rad, parameters[1]*u.rad, self.matched_catalog_stars['RA'], self.matched_catalog_stars['DEC']).to(u.deg)
    #     fig, axs = plt.subplots(2, 2, dpi=dpi)
    #     for ax in axs.flatten():
    #         ax.set_xlabel('theta to center [deg]')
    #         ax.set_xlim(0, 90)

    #     axs[0, 0].scatter(catalog_theta.to(u.deg),
    #                       ang_sep.to(u.arcmin), **kwargs)
    #     axs[0, 0].set_ylabel('angular sepration [arcmin]')
    #     axs[0, 1].scatter(catalog_theta.to(u.deg), xy_sep, **kwargs)
    #     axs[0, 1].set_ylabel('xy sepration [pixel]')
    #     axs[1, 0].scatter(catalog_theta.to(u.deg), theta_rs, **kwargs)
    #     axs[1, 0].set_ylabel('theta res [arcmin]')
    #     axs[1, 1].scatter(catalog_theta.to(u.deg), pa_rs, **kwargs)
    #     axs[1, 1].set_ylabel('position angle res [arcmin]')
    #     axs[0, 0].set_ylim(0, 30)
    #     axs[0, 1].set_ylim(0, 20)
    #     axs[1, 0].set_ylim(-25, 25)
    #     axs[1, 1].set_ylim(-30, 30)

    #     plt.tight_layout()
    #     plt.show()

    def lens_optimize(self, az_alt_range=3, roll_range=4, f_range=1, k_range=0.2):
        az_alt_range = az_alt_range/180*np.pi
        roll_range = roll_range/180*np.pi

        def star_rms(parameters):
            """
            x = [az,alt,roll,f,k]
            """
            stars_az, stars_alt, theta, pa = self.xy_to_az(c_az=parameters[0], c_alt=parameters[1],
                                                           x=self.matched_stars_xy[
                'xcentroid'], y=self.matched_stars_xy['ycentroid'],
                c_x=self.c_x, c_y=self.c_y,
                roll=parameters[2], f=parameters[3], k=parameters[4])
            stars_skycoords = SkyCoord(
                az=stars_az, alt=stars_alt, frame=self.az_frame)
            _, ang_sep, _ = self.matched_catalog_skycoords.match_to_catalog_sky(
                stars_skycoords)
            ang_sep = ang_sep.to(u.arcmin).value
            return np.sqrt(np.mean(ang_sep**2))
        init = np.asarray([self.az, self.alt, self.az_roll, self.f, self.k])
        ranges = np.asarray([az_alt_range, az_alt_range,
                            roll_range, f_range, k_range])
        bonds = np.vstack([init-ranges/2, init+ranges/2]).T
        result = minimize(fun=star_rms, x0=init, bounds=bonds)
        # parameters = result.x
        # self.draw_residual(parameters)
        return init, result

    def plate_optimize(self, ra_dec_range=8, roll_range=8, c_xy_range=200):
        ra_dec_range = ra_dec_range/180*np.pi
        roll_range = roll_range/180*np.pi

        def star_rms(parameters):
            """
            parameters = [ra,dec,roll]
            """
            stars_ra, stars_dec, theta, pa = self.xy_to_eq(
                self.matched_stars_xy['xcentroid'], self.matched_stars_xy['ycentroid'],
                c_ra=parameters[0], c_dec=parameters[1], roll=parameters[2],
                f=self.f, k=self.k,
                c_x=parameters[3], c_y=parameters[4])
            stars_skycoords = SkyCoord(
                ra=stars_ra, dec=stars_dec, frame='icrs')
            idx, d2d, d3d = self.matched_catalog_skycoords.match_to_catalog_sky(
                stars_skycoords)
            d2d = d2d.to(u.arcmin).value
            return np.sqrt(np.mean(d2d**2))

        result = minimize(fun=star_rms, x0=np.asarray([self.ra, self.dec, self.roll, self.c_x, self.c_y]), bounds=(
            (self.ra-ra_dec_range/2, self.ra+ra_dec_range/2),
            (self.dec-ra_dec_range/2, self.dec+ra_dec_range/2),
            (self.roll-roll_range/2, self.roll+roll_range/2),
            (self.c_x-c_xy_range/2, self.c_x+c_xy_range/2),
            (self.c_y-c_xy_range/2, self.c_y+c_xy_range/2))
        )

        return [self.ra, self.dec, self.roll], result

    # def distort_optimize(self):
    #     catalog_theta = angular_separation(
    #         self.ra*u.rad, self.dec*u.rad, self.matched_catalog_skycoords.ra, self.matched_catalog_skycoords.dec)
    #     x = self.matched_stars_xy['xcentroid']
    #     y = self.matched_stars_xy['ycentroid']
    #     _, _, matched_stars_theta, _ = self.xy_to_eq(
    #         self.matched_stars_xy['xcentroid'], self.matched_stars_xy['ycentroid'], c_ra=self.ra, c_dec=self.dec, roll=self.roll, f=self.f, k=self.k, c_x=self.c_x, c_y=self.c_y)
    #     delta_x = x - self.c_x
    #     delta_y = y - self.c_y
    #     delta_xy = np.asarray([delta_x, delta_y])
    #     delta_xy = np.dot([[1, 0], [0, -1]], delta_xy)
    #     r = np.sqrt(delta_x**2+delta_y**2)*self.pixel_size

    #     def func(r, a, b, c, d, e):
    #         return a*r**5+b*r**4+c*r**3+d*r**2+e*r
    #     [self.a, self.b, self.c, self.d, self.e], _ = curve_fit(
    #         func, r, (matched_stars_theta-catalog_theta).to(u.rad))

    #     return self.a, self.b, self.c, self.d, self.e

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
