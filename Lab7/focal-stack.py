import numpy as np
import scipy
import scipy.ndimage
import cv2
import imageio
import argparse
import sys
import multiprocessing
from common.image import *
from common.show import *

parser = argparse.ArgumentParser(description='Extracts a focal stack from a light field.')
parser.add_argument('image_path', metavar="IMAGE", type=str, help="Path to source image file.")
parser.add_argument('n', metavar="N", type=int, help="The dimentions of the lightfield.")
parser.add_argument('-a', '--aperture', metavar="RADIUS", type=float, default=4.5, help="Aperture radius.")
parser.add_argument('-s', '--shift', metavar="AMOUNT", type=float, default=0, help="Shift amount (in px per lightfield dims), for adjusting focal plane.")
parser.add_argument('-g', '--gif', nargs=4, type=float, default=0, help="GIF mode. Parameters: FPS TIME START_SHIFT END_SHIFT")
parser.add_argument('output_path', metavar="OUTPUT", type=str,
                    help="Path where the output image will be stored.")

try:
    args = parser.parse_args()
except SystemExit:
    sys.exit(0)
    
N = args.n
NR = int((N-1)/2)

# Load the image
image = scipy.ndimage.imread(args.image_path)[:,:,0:3]/255.0

# Cut the image into blocks
blocks = image_split_blocks(image, N, N)
print(blocks.shape)

kernel = img_gen(KernelGenCircle(args.aperture, NR), (N,N,1))[:,:,0]
print("Aperture kernel:")
show(kernel)
kernel = kernel[:, :, np.newaxis, np.newaxis, np.newaxis]

blocks = blocks*kernel;

def gather(blocks, shift):
    print("Gathering for shift %f" % shift)
    # Perform shifting for focal adjustment
    for y in range(N):
        for x in range(N):
            if kernel[y,x] < 0.001:
                continue
            dy, dx = y - NR, x - NR
            translate = np.array([dx * shift/(N-1), dy * shift/(N-1)])
            # blocks[y,x] = img_transform_translate(blocks[y,x], -translate, blocks[x,y].shape, order=1)
            blocks[y,x] = scipy.ndimage.interpolation.shift(blocks[y,x], (dy * shift/(N-1), dx * shift/(N-1), 0), order=1)
            
    blocks = blocks.sum(axis=1).sum(axis=0)[np.newaxis, np.newaxis, :, :, :]/(kernel.sum())
    
    return image_join_blocks(blocks)

if args.gif is 0:
    image2 = gather(blocks, args.shift)
             
    # Save the result
    scipy.misc.imsave(args.output_path, image2)

else:
    fps, time, shift_start, shift_end = args.gif
    nframes = fps*time
    frames = np.linspace(shift_start, shift_end, nframes).tolist()
    print("Computing %d frames" % nframes)

    def do_frame(x):
        i, shift = x
        return np.asarray(gather(blocks.copy(), shift)).copy()
    
    pool = multiprocessing.Pool(8)
    
    results = pool.imap(do_frame, enumerate(frames))

    print("Exporting")
    kargs = { 'duration': 1.0/fps }
    imageio.mimsave(args.output_path, results, 'GIF', **kargs)
