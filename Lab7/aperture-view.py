import numpy as np
import scipy
import scipy.ndimage
import cv2
import imageio
import argparse
import sys
from common.image import *
from common.show import *

parser = argparse.ArgumentParser(description='Creates aperture view for a lightfield.')
parser.add_argument('image_path', metavar="IMAGE", type=str, help="Path to source image file.")
parser.add_argument('n', metavar="N", type=int, help="The dimentions of the lightfield.")
parser.add_argument('output_path', metavar="OUTPUT", type=str,
                    help="Path where the output image will be stored.")

try:
    args = parser.parse_args()
except SystemExit:
    sys.exit(0)
    
N = args.n

# Load the image
image = scipy.ndimage.imread(args.image_path)[:,:,0:3]/255.0

# Cut the image into blocks
blocks = image_split_blocks(image, N, N)
print(blocks.shape)

blocks = blocks.transpose((2, 3, 0, 1, 4))
print(blocks.shape)

blocks = np.pad(blocks, ((0,0), (0,0), (0,1), (0,1), (0,0)), mode='constant')
print(blocks.shape)

image2 = image_join_blocks(blocks)
print(image2.shape)

# Save the result
scipy.misc.imsave(args.output_path, image2)
