# bring in image
# generate kernal
# convolve
# display input image and output

from PIL import Image, ImageDraw
import numpy as np
from math import sqrt, pi, exp

def gauseDist(x,mu,sigma):
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )
def generateGaussianKernal(size=5, sigma=1):
    kern = np.zeros((size,size))
    for i, row in enumerate(kern):
        for j, val in enumerate(row):
            dist = sqrt((j-(2))**2 + (i-(2))**2)
            kern[i,j] = gauseDist(dist, 0, sigma)
    return kern/kern.sum()

def convolve(img, kern):
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            if j>2 and j<(img.shape[0]-2) and i>2 and i<(img.shape[1]-2):
                selected_reg = img[i-2:i+3, j-2:j+3]
                img[i,j] = np.multiply(selected_reg, kern).sum()
    return img

input = Image.open("lowres.jpg")
img_arr = np.array(input)
print(img_arr.shape[0:2])
gray = img_arr[:,:,0]*0.2989 + img_arr[:,:,1]*0.5870 + img_arr[:,:,2]*0.1140
PIL_image = Image.fromarray(gray.astype('uint8'), 'L')
PIL_image.show()
kernal = generateGaussianKernal(5, 10)
blur = convolve(gray, kernal)

print (kernal.sum())
PIL_image = Image.fromarray(blur.astype('uint8'), 'L')
PIL_image.show()
# # input.show()

