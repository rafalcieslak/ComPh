import scipy
import cv2
import glob
from scipy import ndimage
from scipy import misc
from common import *

# HDR Joiner

def load_set_memorial():
    image_path = "data/Memorial_SourceImages/"
    names = [image_path + ("memorial00%d.png" % d) for d in range(61,76+1)]
    images = [scipy.ndimage.imread(name)/255 for name in names]
    images = [gamma_decode(image) for image in images]
    pws = np.arange(0, len(images))
    times = 0.03125 * np.power(2, pws);
    times = 1.0/times
    return images, times

def load_sec(image_path):
    filenames = glob.glob(image_path)
    print(filenames)
    print("Reading images...")
    images = [scipy.ndimage.imread(name)/256 for name in filenames]
    images = [gamma_decode(image) for image in images]
    print("Done.")
    times = np.array([read_exif_exposure_time(name) for name in filenames])
    return images, times
    
#images, times = load_set_memorial()
#images, times = load_sec("data/sec2/*.JPG")
images, times = load_sec("data/set3/*.jpg")

# Guess zoom factor for LDR preview
zoom_factor = 800/np.maximum(images[0].shape[0], images[0].shape[1])
print("Using zoom factor %d" % zoom_factor)

show(times)
print("Merging images...")
hdr = merge_linear_ldrs(images, times)
print("Done.")

# Since image is linear anyway, normalize
hdr = hdr / hdr.max()

saveHDR("hdr.out.hdr", hdr)

small_hdr = np.maximum(zoom_3ch(hdr, zoom_factor), 0.0)+0.00001

interactive = True

needs_recalc = False
scale = 0.0
compression = 1.0
ldr = cv2_shuffle(gamma_encode(tone_map(small_hdr, compression, scale)))

def set_compression(c):
    global compression, needs_recalc
    compression = c/100
    needs_recalc = True

def set_scale(s):
    global scale, needs_recalc
    scale = s/100 - 1.0
    needs_recalc = True
    
cv2.namedWindow('ldr')
cv2.createTrackbar('compression','ldr',100,200,set_compression)
cv2.createTrackbar('scale','ldr',100,200,set_scale)

while(1):
    cv2.imshow('ldr',ldr)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    if interactive and needs_recalc:
        needs_recalc = False
        print("Recalc")
        ldr = cv2_shuffle(gamma_encode(tone_map(small_hdr, compression, scale)))

cv2.destroyAllWindows()
