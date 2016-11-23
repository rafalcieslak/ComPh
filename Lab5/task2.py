import numpy as np
import scipy
import scipy.ndimage
import cv2
import time
import imageio
from common.image import *
from common.show import *

show_unwrapped = False
animate = True
animate_speed = 15.0
target_size = (5,8)

probe1, p1 = imageio.imread("data/1_relighting/grace_probe.hdr"), 0.005
probe2, p2 = imageio.imread("data/1_relighting/rnl_probe.hdr"), 0.05
probe3, p3 = imageio.imread("data/1_relighting/uffizi_probe.hdr"), 0.05
probe4, p4 = imageio.imread("data/1_relighting/beach_probe.hdr"), 0.05

probe, p = probe4, p4
       
envmap_scale = 2.0
probeB = img_transform_unwrap_envmap(probe, envmap_scale)

probeB = cv2_shuffle(probeB)/(probeB.max() * p)

if show_unwrapped:
    cv2.imshow('upwrap', gamma_encode(probeB))
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def decimate(image, target_size):
    decimate_factors = (target_size[0] / image.shape[0],
                        target_size[1] / image.shape[1])
    # image = (image*255).astype(np.int16)
    # return scipy.ndimage.zoom(image, [decimate_factors[0], decimate_factors[1], 1])
    print(image.max())
    res = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    print(res.shape)
    print("===")
    return res

# Load teapot data
light_data = []
for phi in [60,30,0,-30,-60]:
    row = []
    for theta in [0,45,90,135,180,225,270,315]:
        fname = "data/1_relighting/data/teapot_%d_%d.png" % (theta,phi)
        teapot = cv2.imread(fname)[:,:,0]/255
        row += [teapot]
    light_data += [row]
light_data = np.asarray(light_data)

def relight(pattern):
    assert(pattern.shape[0] == target_size[0])
    assert(pattern.shape[1] == target_size[1])
    sum = np.zeros((light_data[0,0].shape[0], light_data[0,0].shape[1], 3))
    for phi in range(0,target_size[0]):
        for theta in range(0,target_size[1]):
            sum += light_data[phi,theta,:,:,None] * pattern[phi,theta]
    sum = sum / ((target_size[0])*(target_size[1]))
    return sum
    return np.einsum('ptc,ptyx->yxc',pattern, light_data) / (target_size[0]*target_size[1])

while(animate):
    shift_amt = (time.time() * animate_speed) % 360.0
    amt = int(shift_amt * envmap_scale)
    probe_sh = np.roll(probeB, -amt, axis=1)
    probe_decimated = decimate(probe_sh, target_size)

    relit = relight(probe_decimated)
    
    print(probe_decimated.max())
        
    cv2.imshow('upwraped', gamma_encode(probe_sh))
    cv2.imshow('decimated', scipy.misc.imresize(gamma_encode(probe_decimated), (250, 400, 3), interp='nearest'))
    cv2.imshow('relit', gamma_encode(relit))
    
    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()
        break
