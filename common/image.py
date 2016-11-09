import scipy
import scipy.ndimage
import numpy as np

# A helper function for merging three single-channel images into an RGB image
def combine_channels(ch1, ch2, ch3):
    return np.array([ch1.T, ch2.T, ch3.T]).T

def image_to_3ch(image):
    return combine_channels(image,image,image)

def per_channel(func, image):
    assert(image.shape[2] == 3)
    r,g,b = image[:,:,0],image[:,:,1],image[:,:,2]
    return combine_channels(func(r), func(g), func(b))

def zoom_3ch(image, factor):
    return per_channel(lambda x : scipy.ndimage.zoom(x, factor, order=1), image)

# Shuffle dimentions to match cv2 color layout
def cv2_shuffle(image):
    return np.array([image.T[2], image.T[1], image.T[0]]).T

# Applies an arbitrary image transformation. The function which is
# passed as the second argument will be called with a long list of
# coordinates. It must return an array of identical size, but it may
# modify the coordinates of each point, to mark that its value shall
# be sampled from specified location. For example, a function which
# adds n to the second column of its input will transform the image so
# that it shifts n pixels downwards. What is cool about this
# implementation is that the called function may batch-process indices,
# which allows to create an arbitrary image warp transformation without
# processing pixels in a loop.
def arbitrary_image_transform(source_image, function, target_shape=None, constant=[0,0,0], order=3):
    if target_shape is None:
        target_shape = source_image.shape
    cx,cy = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]))
    coords = np.stack((cx,cy), axis=2).reshape((-1,2), order='F')
    coords2 = np.fliplr(function(np.fliplr(coords)))
    assert coords.shape == coords2.shape, ("Original coords shape %s is not equal to modified coords shape %s." % (coords.shape, coords2.shape))
    coordsB = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=0)
    coordsG = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=1)
    coordsR = np.pad(coords2, ((0,0),(0,1)), mode='constant', constant_values=2)
    ptsB = scipy.ndimage.map_coordinates(source_image, coordsB.T, order=order, cval=constant[2])
    ptsG = scipy.ndimage.map_coordinates(source_image, coordsG.T, order=order, cval=constant[1])
    ptsR = scipy.ndimage.map_coordinates(source_image, coordsR.T, order=order, cval=constant[0])
    pts = np.vstack((ptsB, ptsG, ptsR)).T
    tshape = (target_shape[1], target_shape[0], target_shape[2])
    pts = pts.reshape(tshape, order='F').transpose((1,0,2))
    # A copy is needed due to a bug in opencv which causes it to
    # incorrectly track the data layout of numpy arrays which are
    # temporarily in an optimized layout
    return pts.copy()

# (n,2) -> (n,3)
def pointlist_to_homog(points):
    return np.hstack([points, np.ones((points.shape[0], 1))])

# (n,3), -> (n,2)
def pointlist_from_homog(points):
    points[:,0] = points[:,0]/points[:,2]
    points[:,1] = points[:,1]/points[:,2]
    return points[:,0:2]
