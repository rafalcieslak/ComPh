import scipy
import scipy.ndimage
import numpy as np
from .show import show
from .image import *

def naiveComposite(bg, fg, mask, x, y):
    mask = img_transform_1ch(mask, lambda data: data + np.array([-x,-y]), bg.shape, constant=False, order=0)
    fg = img_transform(fg, lambda data: data + np.array([-x,-y]), bg.shape, constant=[0,0,0], order=0)
    bg[mask] = 0
    fg[~mask] = 0
    return fg + bg

def dotIm(a,b):
    return (a * b).sum()

L = np.array([[0, -1, 0],
              [-1, 4, -1],
              [0, -1, 0]])
def Lconv(image):
    return per_channel(lambda i: scipy.ndimage.convolve(i, L), image)

def poissonGD(bg, fg, mask, x, y, niter):
    mask = img_transform_1ch(mask, lambda data: data + np.array([-x,-y]), bg.shape, constant=False, order=0)
    if len(bg.shape) > 2 and bg.shape[2] == 3:
        mask = combine_channels(mask,mask,mask)
    fg = img_transform(fg, lambda data: data + np.array([-x,-y]), bg.shape, constant=[0,0,0], order=0)
    template = bg.copy()
    template[mask] = False
    b = Lconv(fg)
    x = template + (fg * mask)
    for i in range(niter):
        r = ( b - Lconv(x) )*mask
        alpha = dotIm(r,r)/dotIm(r,Lconv(r))
        x = x + alpha*r
    return x

def poissonConj(bg, fg, mask, x, y, niter):
    mask = img_transform_1ch(mask, lambda data: data + np.array([-x,-y]), bg.shape, constant=False, order=0)
    if len(bg.shape) > 2 and bg.shape[2] == 3:
        mask = combine_channels(mask,mask,mask)
    fg = img_transform(fg, lambda data: data + np.array([-x,-y]), bg.shape, constant=[0,0,0], order=0)
    template = bg.copy()
    template[mask] = False
    b = Lconv(fg)
    x = template + (fg * mask)
    r = ( b - Lconv(x) ) * mask
    d = r
    for i in range(niter):
        #print("iter " ,i)
        alpha = dotIm(r,r)/dotIm(d,Lconv(d))
        x = x + alpha*d
        rnext = ( r - alpha*Lconv(d) ) * mask
        beta = dotIm(rnext, rnext)/dotIm(r,r)
        r = rnext
        d = ( r + beta * d ) * mask
    return x
