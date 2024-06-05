from utils import FishEyeImage
from astropy.table import Table

# hips_star = Table.read('hip_for_cal.fits')

file = '001_0333'

pic = FishEyeImage( raw_path=file+'.CR3', img_path=file+'.jpg', az_mode=False, anno_mode=True)

pic.solve(solve_size=1000)

pic.constellation(fn=file+'_n.jpg')
