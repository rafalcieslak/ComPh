import numpy as np
import scipy
import scipy.ndimage
import cv2
import argparse
import sys
from common.image import *
from common.show import *
from common.compositing import *

parser = argparse.ArgumentParser(description='Performs image compositing.')
parser.add_argument('bg_path', metavar="BG_IMAGE", type=str, help="Path to background image file.")
parser.add_argument('fg_path', metavar="FG_IMAGE", type=str, help="Path to foreground image file.")
parser.add_argument('mask_path', metavar="MASK_IMAGE", type=str, help="Path to mask image file.")
parser.add_argument('--mode', type=str, choices=['naive', 'poissonGD', 'poissonConj'], default='poissonConj', help='Compositing algorithm to use')
parser.add_argument('--niter', type=int, default=50, help='Number of iterations')
parser.add_argument('--pos', nargs=2, type=int, default=[0,0], help="Position of the fg image within the bg image.")
parser.add_argument('-o', '--output', metavar="OUTPUT", type=str, default='output.png',
                    help="Path where the output image will be stored.")

try:
    args = parser.parse_args()
except SystemExit:
    sys.exit(0)

x = args.pos[0]
y = args.pos[1]

bg = scipy.ndimage.imread(args.bg_path)[:,:,0:3]/255.0
fg = scipy.ndimage.imread(args.fg_path)[:,:,0:3]/255.0
mask = scipy.ndimage.imread(args.mask_path)[:,:,0]/255.0 > 0.5

if args.mode == 'naive':
    print("Using Naive Compositing")
    result = naiveComposite(bg, fg, mask, x, y)
elif args.mode == 'poissonGD':
    print("Using Poisson Gradient Descent")
    result = poissonGD(bg, fg, mask, x, y, args.niter)
elif args.mode == 'poissonConj':
    print("Using Poisson Conjugate")
    result = poissonConj(bg, fg, mask, x, y, args.niter)

scipy.misc.imsave(args.output, result.copy())
