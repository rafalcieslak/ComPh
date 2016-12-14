import scipy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# A helper printer which I use instead of print for numpy arrays.
# I don't know why, but print clearly ignores numpy output settings
# for some (large?) arrays. show() fixes this.
def show(array):
    print(np.array_str(array,suppress_small=True,precision=5,max_line_width=200))


# Verbose aliases for gamma transformation
def gamma_decode(img, gamma=2.2):
    return np.power(img, gamma) 

def gamma_encode(img, gamma=2.2):
    return np.power(img, 1.0/gamma)
    
# A hepler function for displaying images within the notebook.
# It may display multiple images side by side, optionally apply gamma transform, and zoom the image.
def show_image(imglist, c='gray', vmin=0, vmax=1, zoom=1, needs_encoding=False):
    if type(imglist) is not list:
       imglist = [imglist]
    n = len(imglist)
    first_img = imglist[0]
    dpi = 77 # pyplot default?
    plt.figure(figsize=(first_img.shape[0]*zoom*n/dpi,first_img.shape[0]*zoom*n/dpi))
    for i in range(0,n):
        img = imglist[i]
        if needs_encoding:
            img = gamma_encode(img)
        plt.subplot(1,n,i + 1)
        plt.tight_layout()    
        plt.axis('off')
        if len(img.shape) == 2:
           img = np.repeat(img[:,:,np.newaxis],3,2)
        plt.imshow(img, interpolation='nearest')
