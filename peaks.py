from utils import FishEyeImage, tt11, rot
from photutils.detection import find_peaks, DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.coordinates import angular_separation, position_angle
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
import numpy as np
import matplotlib.pyplot as plt

hips_star = Table.read('hip2.fits')

file = '291A2094'

pic = FishEyeImage(file+'.jpg', file+'.CR3')

solution = pic.solve(solve_size=1200)

def get_stars(solution, max_mag, min_mag):
    hips_mag = hips_star[np.logical_and(
        hips_star['Vmag'] <= min_mag, hips_star['Vmag'] >= max_mag)]
    ang_sep_list = []
    ra = solution['RA']/180*np.pi
    dec = solution['Dec']/180*np.pi
    for i in range(len(hips_mag)):
        ang_sep = angular_separation(
            ra, dec, hips_mag[i]['RAdeg']/180*np.pi, hips_mag[i]['DEdeg']/180*np.pi)
        ang_sep_list.append(ang_sep)
    hips_mag.add_column(ang_sep_list, name='seprataion')
    stars_table = hips_mag[hips_mag['seprataion']
                           < 0.45*np.pi]

    pa_list = []
    for i in range(len(stars_table)):
        pa = position_angle(
            ra, dec, stars_table[i]['RAdeg']/180*np.pi, stars_table[i]['DEdeg']/180*np.pi)
        pa_list.append(pa)
    stars_table.add_column(pa_list, name='position_angle')

    return stars_table

def detect_stars(self, res=500):
    stars_xy = Table()
    for i in range(self.raw.shape[0]//res):
        for j in range(self.raw.shape[1]//res):
            data = self.raw[i*res:(i+1)*res, j*res:(j+1)*res]
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            threshold = median + (15 * std)
            # stars_found = find_peaks(data, threshold, box_size=11)
            stars_founder = DAOStarFinder(fwhm=5, threshold=20.*std)
            stars_found = stars_founder(data - median)
            if stars_found is not None:
                # stars_found['x_peak'] += j*res
                # stars_found['y_peak'] += i*res
                stars_found['xcentroid'] += j*res
                stars_found['ycentroid'] += i*res
                stars_xy = vstack([stars_xy, stars_found])
    stars_eq = Table()
    # stars_eq.add_column(stars_xy['peak_value'], name='peak_value')
    return stars_xy
stars_det = detect_stars(pic)
stars_det_bright = stars_det
stars_det_positions = np.transpose(
    (stars_det_bright['xcentroid'], stars_det_bright['ycentroid']))


val_stars = get_stars(solution=solution,max_mag=-1,min_mag=6)
r = tt11(val_stars['seprataion'])/0.006
x = r * np.cos(val_stars['position_angle'])
y = r * np.sin(val_stars['position_angle'])

roll = solution['Roll']/180*np.pi

validate_positions = rot(rot([x, y], np.pi/2), roll)
validate_positions = np.dot([[1, 0], [0, -1]], validate_positions)
validate_positions[0] += (pic.raw.shape[1]/2)-0.5
validate_positions[1] += (pic.raw.shape[0]/2)-0.5
validate_apertures = CircularAperture(validate_positions.T, r=2)
show_data = np.log(pic.raw)
vmin = np.log(np.median(show_data)-0.1*np.std(show_data))
vmax = 1.4*np.log(np.std(show_data))
_ = validate_apertures.plot(color='red', lw=3.0)

plt.imshow(show_data)

validate_apertures = CircularAperture(stars_det_positions, r=2)
validate_apertures.plot(color='blue', lw=1.5)

plt.show()