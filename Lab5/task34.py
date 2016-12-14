import numpy as np
import scipy
import scipy.ndimage
import cv2
import time
import imageio
from common.image import *
from common.show import *

probe1, p1 = imageio.imread("data/1_relighting/grace_probe.hdr"), 0.0001
probe2, p2 = imageio.imread("data/1_relighting/rnl_probe.hdr"), 0.05
probe3, p3 = imageio.imread("data/1_relighting/uffizi_probe.hdr"), 0.05
probe4, p4 = imageio.imread("data/1_relighting/beach_probe.hdr"), 0.05

probe, p = probe1, p1
fps = 30
time = 10

envmap_scale = 2.0
probeB = img_transform_unwrap_envmap(probe, envmap_scale)

probeB = probeB/(probeB.max() * p)
print(probeB.shape)

# Load dataset
image_n = 253
data = []
for i in range(0,image_n):
    data += [gamma_decode(scipy.ndimage.imread("data/knight_kneeling/knight_kneeling_%03d.png" % i)/255)]

images = np.asarray(data);
print(images.shape)

light_dirs = np.loadtxt("data/light_directions.txt", usecols=(1,2,3))
print(light_dirs.shape)

r = np.sqrt(np.power(light_dirs,2).sum(axis=1))
thetas = np.arccos(light_dirs[:,1] / r)
phis = np.arctan2(light_dirs[:,0],light_dirs[:,2])

thetas = 180 - thetas*(180/np.pi)
phis = phis*(180/np.pi) + 180

tps = np.hstack([thetas[:,None], phis[:,None]]) * envmap_scale
# show(tps)

light_intens = np.loadtxt("data/light_intensities.txt", usecols=(1,2,3))

probeB = scipy.ndimage.gaussian_filter(probeB, [20,20,0])

def relight(envmap):
    # envmap = scipy.ndimage.gaussian_filter(envmap, [5,5,0])
    sum = np.zeros_like(images[0])
    for i in range(0,image_n):
        tp = tps[i]
        color = envmap[int(tps[i,0]),int(tps[i,1])]
        t = images[i] * color / light_intens[i]
        sum = sum + t
    sum = sum / image_n
    return sum

frames = []
frameno = time * fps
for f in range(0,frameno):
    print("Frame %d/%d" % (f+1,frameno))
    shift = 360 * f / frameno
    shift = int(shift * envmap_scale)
    envmap = np.roll(probeB, -shift, axis=1)
    res = relight(envmap) * 12
    frames += [res]

print("Exporting")
exportname = "output.gif"
kargs = { 'duration': 1/fps }
imageio.mimsave(exportname, frames, 'GIF', **kargs)
 
#cv2.imshow('map', cv2_shuffle(gamma_encode(scipy.ndimage.gaussian_filter(probeB, [20,20,0]))))
#cv2.imshow('relit', cv2_shuffle(gamma_encode(res)))

#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()
