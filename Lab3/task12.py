from common.show import *
from common.image import *

import scipy
import scipy.ndimage
import scipy
import cv2

img_tram = scipy.ndimage.imread("data/green.png")
img_poster = scipy.ndimage.imread("data/poster.png")
img_poster = img_poster[:,:,0:3]

H = np.array([[ 0.8025,  0.0116,  -78.2148],
              [-0.0058,  0.8346, -141.3292],
              [-0.0006, -0.0002,    1.    ]])

# For no interpolation. use order 0.
# For bilinear, use order 1.
warped_poster = img_transform(img_poster, HomographyApplier(H),
                              target_shape=img_tram.shape, order=1)
mask_poster = img_transform(np.ones_like(img_poster), HomographyApplier(H),
                            target_shape=img_tram.shape)

result = (1-mask_poster)*img_tram + mask_poster*warped_poster

cv2.imshow('result',cv2_shuffle(result))
    
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
