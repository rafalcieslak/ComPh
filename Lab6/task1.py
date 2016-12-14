import numpy as np
import scipy
import scipy.ndimage
import scipy.io
import cv2
import multiprocessing
from common.image import *
from common.show import *

img = scipy.ndimage.imread("data/CodedApertureData/cups_board_inp.bmp")/255

filts = scipy.io.loadmat("data/CodedApertureData/filts/filt_scl04.mat")['filts']
filts = filts[0]

cv2.imwrite('input.png', cv2_shuffle(img) * 255)

def deconv(d):
    i, filt = d
    r = deconvL2_frequency(img, filt, 0.01)
    cv2.imwrite(('result_L2_%02d.png' % i), cv2_shuffle(r) * 255)
    # r = deconv_sps(rgb2Y(img), filt, 0.004)
    # cv2.imwrite(('result_sps_%02d.png' % i), image_to_3ch(r) * 255)
    return r

# pool = multiprocessing.Pool()

# results = pool.map(deconv, enumerate(filts), 1)

results = [deconv(d) for d in enumerate(filts)]
    
