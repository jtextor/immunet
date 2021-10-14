# Script with inference: network output + cells detection

## configuration parameter: Threshold for Laplacian-Of-Gaussian cell detection
log_threshold = 0.07

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import json
import numpy as np
from models import model_for_inference
import tifffile
from csbdeep.utils import normalize
from PIL import Image
from skimage.feature import blob_log
from skimage import draw

Mt = model_for_inference("current_best_model.hdf5")

# Optional: look at model structure
# print(Mt.summary())

x = tifffile.imread("tilecache/2020-01-27-phenotyping-paper-cytoagars/tonsil01/57055,8734/components.tiff",key=range(0,6))
x = normalize( x )
x = np.moveaxis( x, 0, -1 )

dist,psi = Mt.predict( np.array([x]) )

dist = dist[0,:,:,0]
psi = psi[0]

# Output 1: 
# Make PNG visualization of distance prediction and all pseudochannels

a = 50*dist
a[a<0]=0
a[a>255]=255

im = a.astype("uint8")
im = Image.fromarray(im)
im.type="I"
im.save("demo-output/cell-center-distance-prediction.png", "PNG")


for i in range( psi.shape[2] ):
    a = 255*psi[:,:,i]
    a[a<0]=0
    a[a>255]=255

    im = a.astype("uint8")
    im = Image.fromarray(im)
    im.type="I"
    im.save(f"demo-output/pseudochannel-{i}.png", "PNG")

# Output 2:
# Use Laplacian-Of-Gaussian filter to detect cell locations and output 
# locations and phenotypes

cell_locations = [(int(x[0]),int(x[1])) 
        for x in blob_log( dist, min_sigma=3, max_sigma=5, threshold=log_threshold)]

# Add pseudo-channels for phenotype identification
for x in cell_locations:
    rr,cc = draw.circle( x[0], x[1], 2, dist.shape )
    ph = list(x) + [float(np.mean(psi[rr,cc,j])) for j in range(psi.shape[2])] 
    print( "\t".join( ["%g" % i for i in ph] ) )