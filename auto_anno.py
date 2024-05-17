from utils import FishEyeImage
from astropy.table import Table

# hips_star = Table.read('hip_for_cal.fits')

file = '003_0682'

pic = FishEyeImage(file+'.jpg', file+'.CR3')

pic.solve(solve_size=1000)

pic.constellation(fn=file+'_n.jpg')
