#!/usr/bin/env python

## Convert sagittal sections to coronal and horizontal sections

import skimage
from skimage import io
import numpy as np
import os
import re
import logging
import argparse
import sys

# Argument parsing
description = '''
  Convert sagittal sections to coronal and horizontal sections.
  Example: `ssh-all hosts.list "uname -a"`
'''
parser = argparse.ArgumentParser(description=description)
direction_help = '''
    Direction for the generated images. Possible values are 'AP', 'DV'.
'''
parser.add_argument('direction', metavar='DIRECTION', 
                    action='store', choices=['AP', 'DV'], 
                    help=direction_help)
starti_help = '''
    The start z index(inclusive, start with 0) in the direction of generated images, 
    for batch processing.
'''
parser.add_argument('starti', type=int, help=starti_help)
endi_help = '''
    The end z index(inclusive) in the direction of generated images, 
    for batch processing.
'''
parser.add_argument('endi', type=int, help=endi_help)

args = parser.parse_args(sys.argv[1:])

direction = args.direction
starti = args.starti
endi = args.endi

if starti > endi:
    starti, endi = endi, starti

# Path
raw_path = '/Users/bradleyzhou/Projects/image3d/t1-head'
out_path = '/Users/bradleyzhou/Projects/image3d/try2'

if not os.path.exists(out_path):
    os.mkdir(out_path)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(os.path.join(out_path, '%s.log' % __name__))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Prepare sorted list of images
logger.info('Getting and sorting list of TIFF images. Assumes as saggital sections.')
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
logger.info('Probing one image for image dimensions and types')
probe_fn = os.path.join(raw_path, img_list[0])
probe_img = io.imread(probe_fn)

# skimage reads image to (row, col), see also http://scikit-image.org/docs/stable/user_guide/numpy_images.html#coordinate-conventions
ny, nx = probe_img.shape
data_type = probe_img.dtype

# Coronal sections, 'AP', anterior - posterior
nx_AP, ny_AP, nz_AP = nz, ny, nx
logger.info('Calculated dimensions for coronal(A-P) sections: x: %d, y: %d, and %d sections.' %
            (nx_AP, ny_AP, nz_AP))

# Horizontal sections, 'DV', dorsal - ventral
nx_DV, ny_DV, nz_DV = nx, nz, ny
logger.info('Calculated dimensions for horizontal(D-V) sections: x: %d, y: %d, and %d sections.' %
            (nx_DV, ny_DV, nz_DV))

logger.info('Preparing output paths')
out_path_AP = os.path.join(out_path, 'AP')
out_path_DV = os.path.join(out_path, 'DV')

if not os.path.exists(out_path_AP):
    os.mkdir(out_path_AP)

if not os.path.exists(out_path_DV):
    os.mkdir(out_path_DV)

def generateAP(iz):
    logger.info('Generating AP image %d' % iz)

    iz_path = os.path.join(out_path_AP, 'AP-%05d.tif' % iz)
    if os.path.exists(iz_path) and os.path.isfile(iz_path):
        img_AP_i = io.imread(iz_path)
    else:
        img_AP_i = np.zeros((ny_AP, nx_AP), dtype=data_type)
    
    for i_raw, img_raw_fn in enumerate(img_list):
        img_raw_path = os.path.join(raw_path, img_raw_fn)
        img_raw = io.imread(img_raw_path)
        
        logger.info('Re-adapting raw image %d to AP image %d' % (i_raw, iz))
        # The result AP(coronal) image: (row 0, col 0) is top-left
        #   row 0 -> ny_AP(raw ny): dorsal -> ventral
        #   col 0 -> nx_AP(raw nz): lateral 0 (raw 0) -> lateral z (raw nz)
        #   z   0 -> nz_AP(raw nx): anterior -> posterior
        img_AP_i[:, i_raw] = img_raw[:, iz]
    
    logger.info('Writing AP image %d' % iz)
    # compress with zlib when saving, ref:
    # http://scikit-image.org/docs/dev/api/skimage.external.tifffile.html#skimage.external.tifffile.TiffWriter
    io.imsave(iz_path, img_AP_i, compress=6)

def generateDV(iz):    
    logger.info('Generating DV image %d' % iz)

    iz_path = os.path.join(out_path_DV, 'DV-%05d.tif' % iz)
    if os.path.exists(iz_path) and os.path.isfile(iz_path):
        img_DV_i = io.imread(iz_path)
    else:
        img_DV_i = np.zeros((ny_DV, nx_DV), dtype=data_type)
    
    for i_raw, img_raw_fn in enumerate(img_list):
        img_raw_path = os.path.join(raw_path, img_raw_fn)
        img_raw = io.imread(img_raw_path)
        
        logger.info('Re-adapting raw image %d to DV image %d' % (i_raw, iz))
        # img_DV_i[ny_DV - i_raw - 1, :] = img_raw[iz, :]
        # The result DV(horizontal) image: (row 0, col 0) is top-left
        #   row 0 -> ny_DV(raw nz): lateral 0 (raw 0) -> lateral z (raw nz)
        #   col 0 -> nx_DV(raw nx): anterior -> posterior
        #   z   0 -> nz_DV(raw ny): dorsal -> ventral
        img_DV_i[i_raw, :] = img_raw[iz, :]
    
    logger.info('Writing DV image %d' % iz)
    # compress with zlib when saving, ref:
    # http://scikit-image.org/docs/dev/DVi/skimage.external.tifffile.html#skimage.external.tifffile.TiffWriter
    io.imsave(iz_path, img_DV_i, compress=6)

if direction.upper() == 'AP':
    if starti < 0:
        starti = 0
    if endi > nz_AP - 1:
        endi = nz_AP - 1
    logger.info('Generating re-sliced images AP(coronal), from %d to %d' % 
                (starti, endi))
    for i_AP in xrange(starti, endi + 1):
        generateAP(i_AP)
elif direction.upper() == 'DV':
    if starti < 0:
        starti = 0
    if endi > nz_DV - 1:
        endi = nz_DV - 1
    logger.info('Generating re-sliced images DV(horizontal), from %d to %d' %
                (starti, endi))
    for i_DV in xrange(starti, endi + 1):
        generateDV(i_DV)
