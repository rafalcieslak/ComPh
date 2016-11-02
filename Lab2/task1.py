import scipy
import cv2
from scipy import ndimage
from scipy import misc
from common import *

## Focal stack

display_result = True
display_contributions = True
zoom_factor = 2

image_dir = "data/focalstack/"
image_files = ["stack1.png", "stack2.png", "stack3.png"]

images = [scipy.ndimage.imread(image_dir + image_file) for image_file in image_files]

result, contributions = focal_stack(images)

scipy.misc.imsave("focalstack.out.png", result)

result = scipy.ndimage.zoom(result, zoom_factor)

if display_result:
    cv2.imshow('result', result)
if display_contributions:
    cv2.imshow('contributions', contributions)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
