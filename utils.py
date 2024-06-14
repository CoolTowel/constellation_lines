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
from scipy.optimize import minimize, curve_fit, differential_evolution
from astropy.time import Time
import json

def rad_convertor(rad):
    return np.arctan2(np.sin(rad),np.cos(rad))

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
    def __init__(self, raw_path, loc, img_path =None, results_path='./results/', az_mode=True, anno_mode=False,
                 raw_iso_corr=False,
                 f=14.6, k=-0.2001, pixel_size=0.006, sensor='full_frame',
                 star_catalog='HIP2_rad.fits', mag_limit=6.5):
        self.results_path = results_path
        self.loc = loc
        self.az_mode = az_mode
        self.raw_path = raw_path
        self.anno_mode = anno_mode
        with rawpy.imread(raw_path) as rawfile:
            raw = rawfile.postprocess(
                gamma=(1, 1), no_auto_bright=True, output_bps=16)[16:4016, 20:6020]

        if img_path is not None:
            self.img = Image.open(img_path)
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
        self.az_frame = AltAz(obstime=self.obstime, location=self.loc, pressure=alt2pres(
                self.loc.height.value), temperature=0*u.deg_C, relative_humidity=0.5, obswl=550*u.nm)
        if not az_mode:
            self.frame = ICRS
        elif az_mode:
            self.frame = self.az_frame
            self.catalog_skycoords = self.catalog_skycoords.transform_to(
                self.frame)
        above_horizen_mask = self.catalog_skycoords.transform_to(self.az_frame).alt>5*u.deg
        self.catalog_skycoords = self.catalog_skycoords[above_horizen_mask]
        ks = [-0.008468431164413271, 0.020895730775804096, -0.013284535904190178, 0.001981509862907541]
        self.lens_para = {'f': f, 'k': k, 'ks':np.zeros(4)}
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
            if not self.anno_mode:
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

    def xmatch(self, sub_region_size=500, sigma=15, sep_limit=35):
        try:
            print('Using existing star detection data')
            self.detected_stars = Table.read(self.results_path+self.raw_path+'.detected_stars'+'_'+str(
                sub_region_size)+'_size_'+str(sigma)+'_sigma.fits')
        except FileNotFoundError:
            self.detected_stars = Table()
            for i in range(self.height//sub_region_size):
                for j in range(self.width//sub_region_size):
                    data = self.raw[i*sub_region_size:(
                        i+1)*sub_region_size, j*sub_region_size:(j+1)*sub_region_size]
                    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                    threshold = median + (sigma * std)
                    stars_founder = DAOStarFinder(fwhm=5, threshold=threshold, min_separation=5)
                    stars_found = stars_founder(data)
                    if stars_found is not None:
                        stars_found['xcentroid'] += j*sub_region_size
                        stars_found['ycentroid'] += i*sub_region_size
                        self.detected_stars = vstack([self.detected_stars, stars_found])
            self.detected_stars.write(self.results_path+self.raw_path+'.detected_stars'+'_'+str(
                sub_region_size)+'_size_'+str(sigma)+'_sigma.fits', format='fits')

        im_u = self.detected_stars['xcentroid']
        im_v = self.detected_stars['ycentroid']
        x, y = self.uv2xy(im_u, im_v)
        star_lon, star_lat, _, _ = self.xy2wcs(x, y)

        detected_star_skycoords = SkyCoord(star_lon, star_lat, frame=self.frame)

        idx_c2s, sep_c2s, _ = self.catalog_skycoords.match_to_catalog_sky( # catalog to star match
            detected_star_skycoords)
        idx_s2c, sep_s2c, _ = detected_star_skycoords.match_to_catalog_sky( # star to catalog match
            self.catalog_skycoords)
        
        idx_s2c_sort_by_idx_c2s = idx_s2c[idx_c2s]
        xmask = idx_s2c_sort_by_idx_c2s==np.indices(idx_c2s.shape).flatten()
        sep_mask = sep_c2s<sep_limit*u.arcmin
        mask = xmask & sep_mask
        idx_c2s_xmatch = idx_c2s[mask]
        self.matched_catalog_skycoords = self.catalog_skycoords[mask]

        if self.az_mode:
            self.catalog_lon = self.matched_catalog_skycoords.az
            self.catalog_lat = self.matched_catalog_skycoords.alt
        else:
            self.catalog_lon = self.matched_catalog_skycoords.ra
            self.catalog_lat = self.matched_catalog_skycoords.dec
        # self.star_skycoords = detect_star_skycoords[idx_c2s_xmatch]
        self.stars_uv = np.asarray(
            [self.detected_stars['xcentroid'][idx_c2s_xmatch], self.detected_stars['ycentroid'][idx_c2s_xmatch]])
        return np.sqrt(np.mean((sep_c2s[mask].to(u.arcmin))**2))

    def lens_func(self, theta, lens_para=None):  # see https://ptgui.com/support.html#3_28
        if lens_para == None:
            lens_para = self.lens_para
        f = lens_para['f']
        k = lens_para['k']
        ks = lens_para['ks']
        r = f*np.sin(k*theta)/k
        return r

    def reverse_lens_func(self, r, lens_para=None):
        if lens_para == None:
            lens_para = self.lens_para

        f = lens_para['f']
        k = lens_para['k']
        ks = lens_para['ks']
        theta = np.arcsin(r*k/f)/k
        return theta-theta*(ks[0]*theta+ks[1]*theta**3+ks[2]*theta**5+ks[3]*theta**7)

    def uv2xy(self, im_u, im_v, plat_para=None):
        if plat_para == None:
            plat_para = self.plat_para
        roll = plat_para['roll']
        cu = plat_para['cu']
        cv = plat_para['cv']
        x = im_u - cu
        y = im_v - cv
        xy = np.asarray([x, y])
        xy = np.dot([[1, 0], [0, -1]], xy)
        xy = rot(xy, -roll - np.pi/2)
        return xy[0], xy[1]

    def xy2uv(self, x, y, plat_para=None):
        if plat_para == None:
            plat_para = self.plat_para
        roll = plat_para['roll']
        cu = plat_para['cu']
        cv = plat_para['cv']
        xy = np.asarray([x, y])
        xy = rot(xy, roll + np.pi/2)
        xy = np.dot([[1, 0], [0, -1]], xy)
        im_u = xy[0] + cu
        im_v = xy[1] + cv
        return im_u, im_v

    def xy2wcs(self, x, y, plat_para=None, lens_para=None):
        if plat_para == None:
            plat_para = self.plat_para
        r = np.sqrt(x**2+y**2)*self.pixel_size
        pa = np.arctan2(y,x)
        theta = self.reverse_lens_func(r=r, lens_para=lens_para)
        if self.az_mode:
            lon, lat = offset_by(
                plat_para['lon'], plat_para['lat'], -pa, theta)
        else:
            lon, lat = offset_by(plat_para['lon'], plat_para['lat'], pa, theta)
        return lon, lat, theta*u.rad, pa*u.rad

    def wcs2xy(self, lon, lat, plat_para=None, lens_para=None):
        if plat_para == None:
            plat_para = self.plat_para
        theta = angular_separation(
            plat_para['lon']*u.rad, plat_para['lat']*u.rad, lon, lat)
        pa = position_angle(plat_para['lon']*u.rad,
                            plat_para['lat']*u.rad, lon, lat)
        pa = rad_convertor(pa)
        r = self.lens_func(theta.to(u.rad).value, lens_para)
        r_pixel = r/self.pixel_size
        if self.az_mode:
            pa = -pa
            x = r_pixel * np.cos(pa)
            y = r_pixel * np.sin(pa)
        else:
            x = r_pixel * np.cos(pa)
            y = r_pixel * np.sin(pa)
        return x, y, theta, pa
    
    def residual(self, plat_para=None, lens_para=None):
        if plat_para == None:
            plat_para = self.plat_para
        if lens_para == None:
            lens_para = self.lens_para
        star_x, star_y = self.uv2xy(im_u=self.stars_uv[0], im_v=self.stars_uv[1], plat_para=plat_para)
        lon, lat, star_theta, star_pa = self.xy2wcs(x=star_x, y=star_y,plat_para = plat_para,lens_para=lens_para)
        stars_skycoords = SkyCoord(lon, lat, frame=self.frame)
        _, ang_sep, _ = self.matched_catalog_skycoords.match_to_catalog_sky(
            stars_skycoords)
        catalog_x, catalog_y, catalog_theta, catalog_pa = self.wcs2xy(
            lon=self.catalog_lon, lat=self.catalog_lat,plat_para = plat_para,lens_para=lens_para)
        xy_sep = np.sqrt((star_x-catalog_x)**2+(star_y-catalog_y)**2)

        theta_res = star_theta-catalog_theta
        pa_res = star_pa-catalog_pa

        return catalog_theta, ang_sep, theta_res, pa_res

    def draw_residual(self, dpi, **kwargs):
        # im_u = self.stars_uv[0]
        # im_v = self.stars_uv[1]
        # star_x, star_y = self.uv2xy(im_u, im_v)
        # lon, lat, star_theta, star_pa = self.xy2wcs(star_x, star_y)
        # stars_skycoords = SkyCoord(lon, lat, frame=self.frame)
        # _, ang_sep, _ = self.catalog_skycoords.match_to_catalog_sky(
        #     stars_skycoords)

        _, _, _, catalog_pa = self.wcs2xy(
            self.catalog_lon, self.catalog_lat)
        # xy_sep = np.sqrt((star_x-catalog_x)**2+(star_y-catalog_y)**2)

        # theta_rs = star_theta-catalog_theta
        # pa_rs = star_pa-catalog_pa
        catalog_theta, ang_sep, theta_res, pa_res = self.residual()
        # catalog_theta = angular_separation(
        #     parameters[0]*u.rad, parameters[1]*u.rad, self.matched_catalog_stars['RA'], self.matched_catalog_stars['DEC']).to(u.deg)
        fig, axs = plt.subplots(2, 2, dpi=dpi)
        for ax in axs.flatten():
            ax.set_xlabel('theta to center [deg]')
            ax.set_xlim(0, 90)

        axs[0, 0].scatter(catalog_theta.to(u.deg),
                          ang_sep.to(u.arcmin), **kwargs)
        axs[0, 0].set_title('{} stars RMS {:.2f}'.format(len(ang_sep),np.sqrt(np.mean((ang_sep.to(u.arcmin))**2))))
        axs[0, 0].set_ylabel('angular sepration [arcmin]')
        axs[0, 1].scatter(catalog_pa.to(u.deg), pa_res.to(u.arcmin), **kwargs)
        axs[0, 1].set_xlabel('potision angle [deg]')
        axs[0, 1].set_ylabel('position angle res [arcmin]')
        axs[0, 1].set_xlim(-180, 180)
        axs[1, 0].scatter(catalog_theta.to(u.deg),
                          theta_res.to(u.arcmin), **kwargs)
        axs[1, 0].set_ylabel('theta res [arcmin]')
        axs[1, 1].scatter(catalog_theta.to(u.deg),
                          pa_res.to(u.arcmin), **kwargs)
        axs[1, 1].set_ylabel('position angle res [arcmin]')
        axs[0, 0].set_ylim(0, 30)
        axs[0, 1].set_ylim(0, 20)
        axs[1, 0].set_ylim(-25, 25)
        axs[1, 1].set_ylim(-30, 30)

        plt.tight_layout()
        plt.show()

    def optimize(self, coord_range=10,  roll_range = 3, f_range=1, k_range=0.2, uv_range=80, minmize_func = minimize):
        coord_range = coord_range/180*np.pi
        roll_range = roll_range/180*np.pi
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
            lens_para = {
            'f': x[5],
            'k': x[6],
            'ks':self.lens_para['ks']}
            im_u = self.stars_uv[0]
            im_v = self.stars_uv[1]
            x, y = self.uv2xy(im_u, im_v, plat_para=plat_para)
            lon, lat, _, _ = self.xy2wcs(x, y, plat_para=plat_para, lens_para= lens_para)
            stars_skycoords = SkyCoord(lon, lat, frame=self.frame)
            _, ang_sep, _ = self.matched_catalog_skycoords.match_to_catalog_sky(
                stars_skycoords)
            ang_sep = ang_sep.to(u.arcmin).value
            return np.mean(ang_sep**2)
        init = np.asarray([self.plat_para['lon'], self.plat_para['lat'],
                           self.plat_para['roll'], self.plat_para['cu'], self.plat_para['cv'],self.lens_para['f'],self.lens_para['k']])
        ranges = np.asarray(
            [coord_range, coord_range, roll_range, uv_range, uv_range, f_range,k_range])
        bounds = np.vstack([init-ranges/2, init+ranges/2]).T
        result = minmize_func(rms1, x0=init, bounds=bounds)
        x = result.x
        self.plat_para['lon'] = x[0]
        self.plat_para['lat'] = x[1]
        self.plat_para['roll'] = x[2]
        self.plat_para['cu'] = x[3]
        self.plat_para['cv'] = x[4]
        self.lens_para['f'] = x[5]
        self.lens_para['k'] = x[6]
        return result
    
    def optimize2(self, angle_range=5, f_range=1, k_range=0.2, uv_range=200):
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
            _,_,_,pa_res = self.residual(plat_para=plat_para)
            return np.mean(pa_res**2)
        init = np.asarray([self.plat_para['lon'], self.plat_para['lat'],
                           self.plat_para['roll'], self.plat_para['cu'], self.plat_para['cv']])
        ranges = np.asarray(
            [angle_range, angle_range, angle_range, uv_range, uv_range])
        bounds = np.vstack([init-ranges/2, init+ranges/2]).T
        result = minimize(rms1, x0=init, bounds=bounds)
        x = result.x
        self.plat_para['lon'] = x[0]
        self.plat_para['lat'] = x[1]
        self.plat_para['roll'] = x[2]
        self.plat_para['cu'] = x[3]
        self.plat_para['cv'] = x[4]
        return result
    
    def outlier_cliping(self, clip_data, theta_range, bin_n, sigma):
        theta_range = theta_range*u.deg
        theta_steps = np.linspace(theta_range[0],theta_range[1],bin_n+1)
        
        catalog_theta , ang_sep, theta_res, pa_res = self.residual()
        star_theta = catalog_theta+theta_res

        mask = np.full(ang_sep.shape, False)
        out_range_mask = (star_theta<theta_range[0]) | (star_theta >= theta_range[1])

        if clip_data == 'a_sep':
            res = ang_sep
        elif clip_data == 'pa':
            res = pa_res

        for i in range(bin_n):
            bin_start = theta_steps[i]
            bin_stop = theta_steps[i+1]
            ang_mask = (star_theta>=bin_start) & (star_theta<bin_stop)
            res_in_bin = res[ang_mask]
            mean = np.mean(res_in_bin)
            sd = np.std(res_in_bin)
            theta_mask = (res>(mean-sigma*sd)) & (res<(mean+sigma*sd))
            mask = (ang_mask & theta_mask) | mask

        mask = mask | out_range_mask
        self.matched_catalog_skycoords = self.matched_catalog_skycoords[mask]
        self.catalog_lon = self.catalog_lon[mask]
        self.catalog_lat = self.catalog_lat[mask]
        self.stars_uv = self.stars_uv[:,mask]

    def distort_optimize(self):
        # only plat optimized, do the radial distortion correction
        catalog_theta , _, theta_res, _ = self.residual()
        star_theta = catalog_theta+theta_res

        def func(theta, k1, k2,k3,k4):
            return theta*(k1*theta+k2*theta**3+k3*theta**5+k4*theta**7)
        
        [k1,k2,k3,k4], _ = curve_fit(
            func, star_theta, theta_res)
        self.lens_para['ks'] = [k1,k2,k3,k4]

        return k1,k2,k3,k4
    
    def spline_distortion_correction(self,resolution=20):
        [sample_u,sample_v]=np.mgrid[0:self.height//resolution,0:self.width//resolution]*resolution+(resolution/2-0.5)
        spamle_x,sample_y = self.uv2xy(sample_u.flatten(),sample_v.flatten())
        sample_lon,sample_lat,_,_ = self.xy2wcs(sample_lon,sample_lat)
        sample_skycoords = SkyCoord(sample_lon,sample_lat,frame=self.frame)


    def constellation(self, fn='test.jpg', cons_file_path='conslines.npy'):
        cons_lines = np.load(cons_file_path)
        draw = ImageDraw.Draw(self.img)

        for i in range(cons_lines.shape[0]):
            star_1 = cons_lines[i][0]
            star_2 = cons_lines[i][1]
            x1,y1,angular_separation1,_ = self.wcs2xy(star_1[0]*u.deg,star_1[1]*u.deg)
            x2,y2,angular_separation2,_ = self.wcs2xy(star_2[0]*u.deg,star_2[1]*u.deg)
            u1,v1 = self.xy2uv(x1,y1)
            u2,v2 = self.xy2uv(x2,y2)

            if angular_separation1 < 0.45*np.pi*u.rad or angular_separation2 < 0.45*np.pi*u.rad:
                if u2 < u1:
                    u1, u2 = u2, u1
                    v1, v2 = v2, v1
                k = (v2-v1)/(u2-u1)
                break_for_star = 20  # 星座连线断开，露出恒星
                u1 += break_for_star*np.cos(np.arctan(k))
                v1 += break_for_star*np.sin(np.arctan(k))
                u2 -= break_for_star*np.cos(np.arctan(k))
                v2 -= break_for_star*np.sin(np.arctan(k))

                draw.line([u1, v1, u2, v2], fill='white', width=7)
        self.img.save(fn)
