from common.show import *
from common.image import *
from shared import *

import scipy
import scipy.ndimage
import scipy
import cv2

dataset = 'stata'
mark_vertices = True
display_weights = False

## ==================

data = get_dataset(dataset)

imgnames = data['imgs']

imgs = [remove_alpha(scipy.ndimage.imread(f)/255) for f in imgnames]
zoom = 1.0
if 'zoom' in data:
    zoom = data['zoom']
points = [np.array(q) for q in data['points']]
print(points[0])
if 'flipxy' in data and data['flipxy']:
    points = [(flip_xy(p1), flip_xy(p2)) for p1,p2 in points]

print("Preparing Hs")
Hs = [np.identity(3)]
for i in range(1,len(imgs)):
    print("Computing H between %d and %d" % (i-1,i))
    Hdiff = find_homography(points[i-1][0], points[i-1][1])
    Habs = Hs[-1] @ Hdiff
    Hs.append(Habs)

center = 0
if 'center' in data:
    center = data['center']
Hci = np.linalg.inv(Hs[center])
Hs = [Hci @ H for H in Hs]
    
His = [np.linalg.inv(H) for H in Hs]

imgBBs = [np.einsum('ba,na->nb', Hi, pointlist_to_homog(image_bounds_pointlist(img)))
          for Hi,img in zip(His, imgs)]

BB = pointlist_from_homog(np.vstack(imgBBs))
show(BB)
xmin, ymin = BB.min(axis=0)
xmax, ymax = BB.max(axis=0)

xsize = int(xmax - xmin)
ysize = int(ymax - ymin)
offset = np.array([xmin, ymin])

result = np.zeros((ysize, xsize, 3))

mask_gen_func = img_gen_mask_smooth

print("Sampling images")
warps = [img_transform_H(              img , H, result.shape, offset=offset)
         for H, img in zip(Hs, imgs)]
print("Sampling masks")
masks = [img_transform_H(mask_gen_func(img) , H, result.shape, offset=offset)
         for H, img in zip(Hs, imgs)]

if display_weights:
    warps = masks

print("Compositing")
total_weight = sum(masks)
total_weight[total_weight < 0.00001] = 1
result = sum([mask*warp for mask,warp in zip(masks,warps)])/total_weight

if mark_vertices:
    for x,y in BB:
        x -= offset[0]
        y -= offset[1]
        cv2.circle(result,(int(x), int(y)),3,(255,0,0),-1)

result = zoom_3ch(result, zoom)
cv2.imshow('result',cv2_shuffle(result))
    
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
