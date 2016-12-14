import numpy as np
import scipy
import scipy.ndimage
import cv2
import imageio
import argparse
import sys
from common.image import *
from common.show import *

parser = argparse.ArgumentParser(description='Extracts a central slice from a light field.')
parser.add_argument('image_path', metavar="IMAGE", type=str, help="Path to source image file.")
parser.add_argument('n', metavar="N", type=int, help="The dimentions of the lightfield.")
parser.add_argument('mode', metavar="M", type=str, help="Mode: h - horizontal, v - vertical.")
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

_, _, w, h, _ = blocks.shape
qmid = int(N/2)

if args.mode == "h":
    mid = int(h/2)
    blocks = blocks[qmid:qmid+1, :, mid:mid+1, :, :]
    zoom = (20,2,1)
elif args.mode == "v":
    mid = int(w/2)
    blocks = blocks[:, qmid:qmid+1, :, mid:mid+1, :]
    zoom = (2,20,1)
else:
    print("Invalid mode selected")
    sys.exit(1)

blocks = blocks.transpose((1,0,2,3,4))
    
print(blocks.shape)

image2 = image_join_blocks(blocks)
print(image2.shape)

# zoom 10 times vertically
image2 = scipy.ndimage.zoom(image2, zoom)

# Save the result
scipy.misc.imsave(args.output_path, image2)
