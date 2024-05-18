from utils import FishEyeImage
from astropy.table import Table

# hips_star = Table.read('hip_for_cal.fits')

file = 'DSC03748'

pic = FishEyeImage(file+'.jpg', file+'.ARW',f=21.5,k=1)

pic.solve(solve_size=1000)

pic.constellation(fn=file+'_n.jpg')
