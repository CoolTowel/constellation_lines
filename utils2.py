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
import rawpy
import exiftool
from photutils.detection import find_peaks, DAOStarFinder
import astropy.units as u
from scipy.optimize import minimize, curve_fit
from astropy.time import Time
import json


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
    def __init__(self, raw_path, loc, results_path='./results/', az_mode=True, raw_iso_corr=False,
                 f=14.6, k=-0.19, pixel_size=0.006, sensor='full_frame',
                 star_catalog='HIP2_rad.fits', mag_limit=6.5):
        self.results_path = results_path
        self.loc = loc
        self.az_mode = az_mode
        self.raw_path = raw_path
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

        # if cutout:
        #     ii = (self.height-cutout)//2
        #     jj = (self.width-cutout)//2
        #     self.raw = self.raw[ii:ii+cutout, jj:jj+cutout]
        #     self.height = self.raw.shape[0]
        #     self.width = self.raw.shape[1]

        if raw_iso_corr:
            self.raw[self.raw < 60000] = self.raw[self.raw <
                                                  60000]//(self.iso/1600)
        if pixel_size is not None:
            self.pixel_size = pixel_size
        else:
            if sensor == 'full_frame':
                self.pixel_size = 24/self.height
            elif sensor == 'apsc':
                self.pixel_size = 24/self.height/1.55

        self.catalog = Table.read(star_catalog)
        self.catalog = self.catalog[
            (self.catalog['Hpmag'] < mag_limit) & (self.catalog['Hpmag'] > 1)]
        self.catalog_skycoords = SkyCoord(
            ra=self.catalog['RA'], dec=self.catalog['DEC'], frame='icrs', unit='rad')
        if not az_mode:
            self.frame = ICRS
        elif az_mode:
            self.frame = AltAz(obstime=self.obstime, location=self.loc, pressure=alt2pres(
                self.loc.height.value), temperature=0*u.deg_C, relative_humidity=0.5, obswl=550*u.nm)
            self.catalog_skycoords = self.catalog_skycoords.transform_to(
                self.frame)
        self.lens_para = {'f': f, 'k': k, 'ks': np.zeros(3)}
        self.plat_para = {
            'lon': 0,
            'lat': 0,
            'roll': 0,
            'cu': self.raw.shape[1]/2,
            'cv': self.raw.shape[0]/2}

    def solve(self, solve_size=800):
        try:
            f = open(self.results_path+self.raw_path+'.solution', "r")
            self.solution = json.loads(f.read())
            f.close()
        except:
            print('using t3')
            import tetra3
            t3 = tetra3.Tetra3()
            ii = (self.height-solve_size)//2
            jj = (self.width-solve_size)//2
            data = self.raw[ii:ii+solve_size, jj:jj+solve_size]
            data4t3 = data.astype(np.uint16)
            data4t3[data4t3 <= 0] = 0
            img4t3 = Image.fromarray(data4t3)
            t3_solution = t3.solve_from_image(img4t3, distortion=-0.0085542)
            ra = float(t3_solution['RA']/180*np.pi)
            dec = float(t3_solution['Dec']/180*np.pi)
            eq_roll = float(t3_solution['Roll']/180*np.pi)
            self.solution = {'ra': ra, 'dec': dec, 'eq_roll': eq_roll}
            f = open(self.results_path+self.raw_path+'.solution', "w")
            f.write(json.dumps(self.solution))
            f.close()

        print(self.solution)
        if not self.az_mode:
            self.plat_para['lon'] = self.solution['ra']
            self.plat_para['lat'] = self.solution['dec']
            self.plat_para['roll'] = self.solution['eq_roll']
        elif self.az_mode:
            center_eq = SkyCoord(ra=self.solution['ra'],
                                 dec=self.solution['dec'],
                                 frame='icrs', unit='rad')
            center_az = center_eq.transform_to(self.frame)
            self.plat_para['lon'] = center_az.az.to('rad').value
            self.plat_para['lat'] = center_az.alt.to('rad').value
            eq_of_north = SkyCoord(ra=0, dec=90, frame='icrs', unit='deg')
            az_of_north = eq_of_north.transform_to(self.frame)
            c = position_angle(center_az.az, center_az.alt,
                               az_of_north.az, az_of_north.alt)
            self.plat_para['roll'] = self.solution['eq_roll']+c.to('rad').value
            # 地平坐标经度自东向西增加，与天球坐标相反，故az坐标下计得所有pa需加负号处理。因此负负得正，此处为相加

    def initial_xmatch(self, sub_region_size=500, sigma=15, sep_limit=30):
        try:
            self.stars_uv = Table.read(self.results_path+self.raw_path+'.stars_uv'+'_'+str(
                sub_region_size)+'_size_'+str(sigma)+'_sigma.fits')
        except FileNotFoundError:
            self.stars_uv = Table()
            for i in range(self.height//sub_region_size):
                for j in range(self.width//sub_region_size):
                    data = self.raw[i*sub_region_size:(
                        i+1)*sub_region_size, j*sub_region_size:(j+1)*sub_region_size]
                    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                    threshold = median + (sigma * std)
                    stars_founder = DAOStarFinder(fwhm=5, threshold=threshold)
                    stars_found = stars_founder(data)
                    if stars_found is not None:
                        stars_found['xcentroid'] += j*sub_region_size
                        stars_found['ycentroid'] += i*sub_region_size
                        self.stars_uv = vstack([self.stars_uv, stars_found])
            self.stars_uv.write(self.results_path+self.raw_path+'.stars_uv'+'_'+str(
                sub_region_size)+'_size_'+str(sigma)+'_sigma.fits', format='fits')

        im_u = self.stars_uv['xcentroid']
        im_v = self.stars_uv['ycentroid']
        x, y = self.uv2xy(im_u, im_v)
        star_lon, star_lat, _, _ = self.xy2wcs(x, y)

        detect_star_skycoords = SkyCoord(star_lon, star_lat, frame=self.frame)

        idx, sep, _ = self.catalog_skycoords.match_to_catalog_sky(
            detect_star_skycoords)
        sep_constraint = sep < sep_limit*u.arcmin
        self.catalog_skycoords = self.catalog_skycoords[sep_constraint]
        matched_idx = idx[sep_constraint]
        self.star_skycoords = detect_star_skycoords[matched_idx]
        self.stars_uv = np.asarray(
            [self.stars_uv['xcentroid'][matched_idx], self.stars_uv['xcentroid'][matched_idx]])
        return np.sqrt(np.mean((sep[sep_constraint].to(u.arcmin))**2))

    def lens_func(self, theta, lens_para=None):  # see https://ptgui.com/support.html#3_28
        if lens_para == None:
            lens_para = self.lens_para
        f = lens_para['f']
        k = lens_para['k']
        r = f*np.sin(k*theta)/k
        return r

    def reverse_lens_func(self, r, lens_para=None):
        if lens_para == None:
            lens_para = self.lens_para
        f = lens_para['f']
        k = lens_para['k']
        return np.arcsin(r*k/f)/k

    def uv2xy(self, u, v, plat_para=None):
        if plat_para == None:
            plat_para = self.plat_para
        x = u - plat_para['cu']
        y = v - plat_para['cv']
        xy = np.asarray([x, y])
        xy = np.dot([[1, 0], [0, -1]], xy)
        xy = rot(xy, -plat_para['roll'] - np.pi/2)
        return xy[0], xy[1]

    def xy2wcs(self, x, y, plat_para=None, lens_para=None):
        if plat_para == None:
            plat_para = self.plat_para
        r = np.sqrt(x**2+y**2)*self.pixel_size
        pa = np.arctan2(x, y)
        pa[pa < 0] += 2*np.pi
        theta = self.reverse_lens_func(r=r, lens_para=lens_para)
        if self.az_mode:
            lon, lat = offset_by(
                plat_para['lon'], plat_para['lat'], -pa, theta)
        else:
            lon, lat = offset_by(plat_para['lon'], plat_para['lat'], pa, theta)
        return lon, lat, theta*u.rad, pa*u.rad

    def wcs2xy(self,lon,lat,plat_para=None, lens_para=None):
        if plat_para == None:
            plat_para = self.plat_para
        theta = angular_separation(plat_para['lon']*u.rad,plat_para['lat']*u.rad,lon,lat)
        pa = position_angle(plat_para['lon']*u.rad,plat_para['lat']*u.rad,lon,lat)
        r = self.lens_func(theta, lens_para)
        r_pixel = r/self.pixel_size
        if self.az_mode:
            pa = -pa
            x = r_pixel * np.cos(pa)
            y = r_pixel * np.sin(pa)/self.pixel_size
        else:
            x = r_pixel * np.cos(pa)/self.pixel_size
            y = r_pixel * np.sin(pa)/self.pixel_size
        return x, y, theta, pa

    def draw_residual(self, dpi, **kwargs):
        im_u = self.stars_uv[0]
        im_v = self.stars_uv[1]
        star_x, star_y = self.uv2xy(im_u, im_v)
        lon, lat, star_theta , star_pa = self.xy2wcs(star_x, star_y)
        stars_skycoords = SkyCoord(lon, lat, frame=self.frame)
        _, ang_sep, _ = self.catalog_skycoords.match_to_catalog_sky(stars_skycoords)

        if self.az_mode:
            catalog_lon = self.catalog_skycoords.az
            catalog_lat = self.catalog_skycoords.alt
            
        catalog_x, catalog_y, catalog_theta, catalog_pa = self.wcs2xy(catalog_lon, catalog_lat)
        xy_sep = np.sqrt((star_x-catalog_x)**2+(star_y-catalog_y)**2)

        theta_rs = star_theta-catalog_theta
        pa_rs = star_pa-catalog_pa

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
        axs[1, 0].scatter(catalog_theta.to(u.deg), theta_rs.to(u.arcmin), **kwargs)
        axs[1, 0].set_ylabel('theta res [arcmin]')
        axs[1, 1].scatter(catalog_theta.to(u.deg), pa_rs.to(u.arcmin), **kwargs)
        axs[1, 1].set_ylabel('position angle res [arcmin]')
        axs[0, 0].set_ylim(0, 30)
        axs[0, 1].set_ylim(0, 20)
        axs[1, 0].set_ylim(-25, 25)
        axs[1, 1].set_ylim(-30, 30)

        plt.tight_layout()
        plt.show()

    def optimize(self, angle_range=5, f_range=1, k_range=0.2, uv_range=200):
        angle_range = angle_range/180*np.pi
        def rms1(x):
            """
            x = [lon,lat,roll,cu,cv,f,k]
            """
            plat_para = {
                'lon': x[0],
                'lat': x[1],
                'roll': x[2],
                'cu': x[3],
                'cv': x[4]}
            im_u = self.stars_uv[0]
            im_v = self.stars_uv[1]
            x, y = self.uv2xy(im_u, im_v, plat_para=plat_para)
            lon, lat, _, _ = self.xy2wcs(x, y, plat_para=plat_para)
            stars_skycoords = SkyCoord(lon, lat, frame=self.frame)
            _, ang_sep, _ = self.catalog_skycoords.match_to_catalog_sky(
                stars_skycoords)
            ang_sep = ang_sep.value
            return np.sqrt(np.mean(ang_sep**2))
        init = np.asarray([self.plat_para['lon'], self.plat_para['lat'],
                self.plat_para['roll'], self.plat_para['cu'], self.plat_para['cv']])
        ranges = np.asarray([angle_range,angle_range,angle_range,uv_range,uv_range])
        bounds = np.vstack([init-ranges/2, init+ranges/2]).T
        result = minimize(fun=rms1, x0=init, bounds=bounds)
        x = result.x
        self.plat_para = {
            'lon': x[0],
            'lat': x[1],
            'roll': x[2],
            'cu': x[3],
            'cv': x[4]}

        # parameters = result.x
        # self.draw_residual(parameters)
        return 0

    def plate_optimize(self, ra_dec_range=8, roll_range=8, cxy_range=200):
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
                cx=parameters[3], cy=parameters[4])
            stars_skycoords = SkyCoord(
                ra=stars_ra, dec=stars_dec, frame='icrs')
            idx, d2d, d3d = self.matched_catalog_skycoords.match_to_catalog_sky(
                stars_skycoords)
            d2d = d2d.to(u.arcmin).value
            return np.sqrt(np.mean(d2d**2))

        result = minimize(fun=star_rms, x0=np.asarray([self.ra, self.dec, self.roll, self.cx, self.cy]), bounds=(
            (self.ra-ra_dec_range/2, self.ra+ra_dec_range/2),
            (self.dec-ra_dec_range/2, self.dec+ra_dec_range/2),
            (self.roll-roll_range/2, self.roll+roll_range/2),
            (self.cx-cxy_range/2, self.cx+cxy_range/2),
            (self.cy-cxy_range/2, self.cy+cxy_range/2))
        )

        return [self.ra, self.dec, self.roll], result

    def distort_optimize(self):
        catalog_theta = angular_separation(
            self.ra*u.rad, self.dec*u.rad, self.matched_catalog_skycoords.ra, self.matched_catalog_skycoords.dec)
        x = self.matched_stars_xy['xcentroid']
        y = self.matched_stars_xy['ycentroid']
        _, _, matched_stars_theta, _ = self.xy_to_eq(
            self.matched_stars_xy['xcentroid'], self.matched_stars_xy['ycentroid'], c_ra=self.ra, c_dec=self.dec, roll=self.roll, f=self.f, k=self.k, cx=self.cx, cy=self.cy)
        delta_x = x - self.cx
        delta_y = y - self.cy
        delta_xy = np.asarray([delta_x, delta_y])
        delta_xy = np.dot([[1, 0], [0, -1]], delta_xy)
        r = np.sqrt(delta_x**2+delta_y**2)*self.pixel_size

        def func(r, a, b, c, d, e):
            return a*r**5+b*r**4+c*r**3+d*r**2+e*r
        [self.a, self.b, self.c, self.d, self.e], _ = curve_fit(
            func, r, (matched_stars_theta-catalog_theta).to(u.rad))

        return self.a, self.b, self.c, self.d, self.e

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
