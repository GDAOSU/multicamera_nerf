import PIL
import numpy as np 
import os
import glob
from PIL import Image

def npy2tiff(in_dir):
    npys=glob.glob(os.path.join(in_dir,'*.npy'))
    for npy in npys:
        name=npy[:-4]
        if os.path.exists(name+'.tif'):
            continue
        depth=np.load(npy)
        depth[depth==0]=np.nan
        img=Image.fromarray(depth)
        img.save(name+'.tif')

npy2tiff(r'J:\xuningli\cross-view\ns\nerfstudio\renders\geomvs_test2_depth')