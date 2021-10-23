# bring in image
# generate kernel
# convolve
# display input image and output

from typing import cast
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from math import sqrt, pi, exp

def gauseDist(x,mu,sigma):
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )
def generateGaussiankernel(size=5, sigma=1):
    kern = np.zeros((size,size))
    for i, row in enumerate(kern):
        for j, val in enumerate(row):
            dist = sqrt((j-(2))**2 + (i-(2))**2)
            kern[i,j] = gauseDist(dist, 0, sigma)
    return kern/kern.sum()

def generateSobelX():
    return np.array([[-1,0,1],[-2, 0, 2],[-1, 0, 1]])
def generateSobelY():
    return np.array([[1,2,1],[0, 0, 0],[-1, -2, -1]])
def convolve(img, kern):
    size = int((kern.shape[1]-1)/2)
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            if j>size and j<(img.shape[0]-size) and i>size and i<(img.shape[1]-size):
                selected_reg = img[i-size:(i+size+1), j-size:(j+size+1)]
                num = np.multiply(selected_reg,kern).sum()
                img[i,j] = num
    return img

input = Image.open("cat.jpg")
gray = np.array(ImageOps.grayscale(input))
print(gray.shape[0:2])
# gray = img_arr[:,:,0]*0.2989 + img_arr[:,:,1]*0.5870 + img_arr[:,:,2]*0.1140
# PIL_image = Image.fromarray(gray.astype('uint8'), 'L')
# PIL_image.show()
kernel = generateGaussiankernel(5, 10)
blur = convolve(gray, kernel)
# kernel = generateSobelY()
# x_sobel = np.convolve(blur, kernel)
# x_sobel = x_sobel-(x_sobel<200)
PIL_image = Image.fromarray(blur.astype('uint8'), 'L')
PIL_image.show()
# # input.show()

