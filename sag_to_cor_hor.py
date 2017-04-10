#!/usr/bin/env python

## Convert sagittal sections to coronal and horizontal sections

import skimage
from skimage import io
import numpy as np
import os
import re

raw_path = '/Users/bradleyzhou/Projects/image3d/t1-head'
out_path = '/Users/bradleyzhou/Projects/image3d/try2'

if not os.path.exists(out_path):
    os.mkdir(out_path)

img_list = [img for img in os.listdir(raw_path) if re.search(r'\d+[.][tT][iI][fF]', img)]

def catchSerial(s):
    serial = re.findall(r'(\d+)[.]', s)
    if serial:
        return int(serial[0])
    return s

img_list.sort(key=catchSerial)

# z: number of sagittal sections in total
nz = len(img_list)

# probe one single image for image dimensions and types
probe_fn = os.path.join(raw_path, img_list[0])
probe_img = io.imread(probe_fn)

# skimage reads image to (row, col), see also http://scikit-image.org/docs/stable/user_guide/numpy_images.html#coordinate-conventions
ny, nx = probe_img.shape
data_type = probe_img.dtype

# Coronal sections, 'AP', anterior - posterior
nx_AP, ny_AP, nz_AP = (nz, ny, nx)

# Horizontal sections, 'DV', dorsal - ventral
nx_DV, ny_DV, nz_DV = (nx, nz, ny)

out_path_AP = os.path.join(out_path, 'AP')
out_path_DV = os.path.join(out_path, 'DV')

if not os.path.exists(out_path_AP):
    os.mkdir(out_path_AP)

if not os.path.exists(out_path_DV):
    os.mkdir(out_path_DV)

for i_raw, img_raw_fn in enumerate(img_list):
    print('Raw image %d' % i_raw)
    img_raw_path = os.path.join(raw_path, img_raw_fn)
    img_raw = io.imread(img_raw_path)
    
    # A-P
    for i_AP in xrange(nz_AP):
        # print('AP image %d' % i_AP)
        i_AP_path = os.path.join(out_path_AP, 'AP-%05d.tif' % i_AP)
        if os.path.exists(i_AP_path) and os.path.isfile(i_AP_path):
            img_AP_i = io.imread(i_AP_path)
        else:
            img_AP_i = np.zeros((ny_AP, nx_AP), dtype=data_type)
        
        # The result AP(coronal) image: (row 0, col 0) is top-left
        #   row 0 -> ny_AP(raw ny): dorsal -> ventral
        #   col 0 -> nx_AP(raw nz): lateral 0 (raw 0) -> lateral z (raw nz)
        #   z   0 -> nz_AP(raw nx): anterior -> posterior
        img_AP_i[:, i_raw] = img_raw[:, i_AP]
        # compress with zlib when saving, ref:
        # http://scikit-image.org/docs/dev/api/skimage.external.tifffile.html#skimage.external.tifffile.TiffWriter
        io.imsave(i_AP_path, img_AP_i, compress=6)
    
    # D-V
    for i_DV in xrange(nz_DV):
        # print('DV image %d' % i_DV)
        i_DV_path = os.path.join(out_path_DV, 'DV-%05d.tif' % i_DV)
        if os.path.exists(i_DV_path) and os.path.isfile(i_DV_path):
            img_DV_i = io.imread(i_DV_path)
        else:
            img_DV_i = np.zeros((ny_DV, nx_DV), dtype=data_type)
        
        # img_DV_i[ny_DV - i_raw - 1, :] = img_raw[i_DV, :]
        # The result DV(horizontal) image: (row 0, col 0) is top-left
        #   row 0 -> ny_DV(raw nz): lateral 0 (raw 0) -> lateral z (raw nz)
        #   col 0 -> nx_DV(raw nx): anterior -> posterior
        #   z   0 -> nz_DV(raw ny): dorsal -> ventral
        img_DV_i[i_raw, :] = img_raw[i_DV, :]
        # compress with zlib when saving, ref:
        # http://scikit-image.org/docs/dev/api/skimage.external.tifffile.html#skimage.external.tifffile.TiffWriter
        io.imsave(i_DV_path, img_DV_i, compress=6)
