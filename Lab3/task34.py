from common.show import *
from common.image import *
from shared import *

import scipy
import scipy.ndimage
import scipy
import cv2

dataset = 'stata'
mark_vertices = True

data = get_dataset(dataset)

img1 = remove_alpha(scipy.ndimage.imread(data['img1'])/255)
img2 = remove_alpha(scipy.ndimage.imread(data['img2'])/255)

H1 = np.identity(3)
H2 = find_homography(data['points1'], data['points2'])
print("H1:")
show(H1)
print("H2:")
show(H2)

H1i = np.linalg.inv(H1)
H2i = np.linalg.inv(H2)

img1_BB = np.einsum('ba,na->nb', H1i, pointlist_to_homog(image_bounds_pointlist(img1)))
img2_BB = np.einsum('ba,na->nb', H2i, pointlist_to_homog(image_bounds_pointlist(img2)))

show(image_bounds_pointlist(img2))
BB = pointlist_from_homog(np.vstack([img1_BB, img2_BB]))
show(BB)
xmin, ymin = BB.min(axis=0)
xmax, ymax = BB.max(axis=0)
print(xmin)
print(xmax)
print(ymin)
print(ymax)

xsize = xmax - xmin
ysize = ymax - ymin
offset = np.array([xmin, ymin])

result = np.zeros((ysize, xsize, 3))
    
warp1 = img_transform_H(             img1 , H1, result.shape, offset=offset)
mask1 = img_transform_H(np.ones_like(img1), H1, result.shape, offset=offset)

warp2 = img_transform_H(             img2 , H2, result.shape, offset=offset)
mask2 = img_transform_H(np.ones_like(img2), H2, result.shape, offset=offset)

total_weight = mask1 + mask2
total_weight[total_weight < 0.00001] = 1
result = (mask1*warp1 + mask2*warp2)/total_weight

if mark_vertices:
    for x,y in BB:
        x -= offset[0]
        y -= offset[1]
        cv2.circle(result,(int(x), int(y)),3,(255,0,0),-1)

cv2.imshow('result',cv2_shuffle(result))
    
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
