import scipy
import numpy as np
import PIL.Image
import cv2
from scipy import ndimage
import scipy.ndimage.filters as ndfilt

# A helper printer which I use instead of print for numpy arrays.
# I don't know why, but print clearly ignores numpy output settings
# for some (large?) arrays. show() fixes this.
def show(array):
    print(np.array_str(array,suppress_small=True,precision=5))
    
# A helper function for merging three single-channel images into an RGB image
def combine_channels(ch1, ch2, ch3):
    return np.array([ch1.T, ch2.T, ch3.T]).T

def per_channel(func, image):
    assert(image.shape[2] == 3)
    r,g,b = image[:,:,0],image[:,:,1],image[:,:,2]
    return combine_channels(func(r), func(g), func(b))

def zoom_3ch(image, factor):
    return per_channel(lambda x : scipy.ndimage.zoom(x, factor, order=1), image)

# Shuffle dimentions to match cv2 color layout
def cv2_shuffle(image):
    return np.array([image.T[2], image.T[1], image.T[0]]).T

# Verbose aliases for gamma transformation
def gamma_decode(img, gamma=2.2):
    return np.power(img, gamma) 

def gamma_encode(img, gamma=2.2):
    return np.power(img, 1.0/gamma)

def read_exif_exposure_time(filename):
    img = PIL.Image.open(filename)
    exif_data = img._getexif()
    num,denom = exif_data[33434]
    return num/float(denom)

def saveHDR(filename, image):
    f = open(filename, "wb")
    f.write(str.encode("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n"))
    f.write(str.encode("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1])))
    
    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)
    
    rgbe.flatten().tofile(f)
    f.close()

def per_range(a):
    high = np.percentile(a,90)
    low = np.percentile(a,10)
    return low,high

def per_range_debug(images):
    show(np.asarray([per_range(q) for q in images]))
    
def compute_gradients(image):
    image_up   = np.pad(image, ((0,1),(0,0)), mode='edge')[ 1:,:].astype(int)
    image_down = np.pad(image, ((1,0),(0,0)), mode='edge')[:-1,:].astype(int)
    image_left = np.pad(image, ((0,0),(0,1)), mode='edge')[:, 1:].astype(int)
    image_right= np.pad(image, ((0,0),(1,0)), mode='edge')[:,:-1].astype(int)
    dh = np.abs(image_up - image_down)
    dv = np.abs(image_left - image_right)
    return (dh+dv)/255.0/2.0

def focal_stack(images):
    print("Computing focal stack from %d images" % len(images))
    grads = [compute_gradients(i) for i in images]
    grads = [np.power(p,3.0) for p in grads]
    grads = [scipy.ndimage.gaussian_filter(p, 0.5) for p in grads]
    total = np.maximum(sum(grads), 0.00000000001)
    perc = [g.astype(float)/total for g in grads]
    contribs = combine_channels(perc[0], perc[1], perc[2])
    res = sum([p*i for p, i in zip(perc, images)])/255.0
    return res, contribs

def image_to_grayscale(image):
    assert(image.shape[2] == 3)
    Y  = np.dot(image, np.array([ 0.2989,  0.5866,  0.1145]))
    return Y

def resp_curve(image, midpoint):
    return np.exp(-4.0*np.power((image - midpoint)/midpoint, 2))

def merge_linear_ldrs(images, times):
    #per_range_debug(images)
    gray_images = [image_to_grayscale(i) for i in images]
    Ilins = [i/t for i,t in zip(gray_images,times)]
    #per_range_debug(Ilins)
    weights = [resp_curve(x, 0.5) for x,time in zip(gray_images, times)]
    wtest = [q[50,50] for q in weights]
    weights = [combine_channels(w,w,w) for w in weights]
    colorIlins = [i/t for i,t in zip(images,times)]
    wtI = [w*t*Ilin for w,t,Ilin in zip(weights,times,colorIlins)]
    wt2 = [w*t*t    for w,t      in zip(weights,times)]
    #wtI = [w*Ilin for w,t,Ilin in zip(weights,times,colorIlins)]
    #wt2 = [w    for w,t      in zip(weights,times)]
    x = sum(wtI)/sum(wt2)
    return x


def tone_map(hdr_image, factor, const):
    intensity = image_to_grayscale(hdr_image)
    r = hdr_image[:,:,0] / intensity
    g = hdr_image[:,:,1] / intensity
    b = hdr_image[:,:,2] / intensity
    log_intensity = np.log10(intensity)
    log_base = cv2.bilateralFilter(log_intensity.astype(np.float32),7,45,45)
    log_detail = log_intensity - log_base
    log_out = log_base * factor + log_detail - const
    r = r * np.power(10, log_out)
    g = g * np.power(10, log_out)
    b = b * np.power(10, log_out)
    out = combine_channels(r,g,b)
    return out
