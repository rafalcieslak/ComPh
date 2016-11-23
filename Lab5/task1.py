import numpy as np
import scipy
import scipy.ndimage
import cv2
from common.image import *
from common.show import *

img_L1 = scipy.ndimage.imread("data/2_photometricstereo/teapot_0_-1_1.png")[:,:,0]/255
img_L2 = scipy.ndimage.imread("data/2_photometricstereo/teapot_-1_1_1.png")[:,:,0]/255
img_L3 = scipy.ndimage.imread("data/2_photometricstereo/teapot_1_1_1.png" )[:,:,0]/255

L1 = np.array([ 0, -1, 1])
L2 = np.array([-1,  1, 1])
L3 = np.array([ 1,  1, 1])

L = np.vstack([L1,L2,L3])

Ns = np.einsum("ab,xyb->xya", np.linalg.inv(L), np.dstack([img_L1, img_L2, img_L3]))
Zs = np.maximum(Ns[:,:,2], 0.001)
Ns = Ns / Zs[:,:,None]

Xs = Ns[:,:,0]
Xs = np.maximum(0, Xs)
Ys = Ns[:,:,1]
Ys = np.maximum(0, Ys)

TRESH = 0.001
mask = (img_L1 > TRESH) | (img_L2 > TRESH) | (img_L3 > TRESH)

res = combine_channels(Xs, Ys, np.zeros_like(Xs)) * mask[:,:,None]

cv2.imshow('normals', cv2_shuffle(res))
    
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
