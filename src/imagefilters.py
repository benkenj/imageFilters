from PIL import Image
import numpy as np
import math
from scipy import signal

# compute a box filter
# input: size of filter N
# output: a 2-d N*N box filter array
def boxfilter(N):
    assert(N % 2 == 1)
    return np.full((N,N),1.0/(float(N)*float(N)))

# perform a gaussian filter on a 1-dimensional array
# input: sigma for standard deviation
# output: a 1-d gaussian filter with standard deviation sigma
def gauss1d(sigma):
    fsigma = float(sigma)
    #compute a normal distribution in a 1-d array
    length = int(math.ceil(fsigma * 6) if math.ceil(fsigma * 6) % 2 == 1 else math.ceil(fsigma * 6) + 1)
    gaus = np.arange(-(length-1)/2, ((length-1)/2)+1, 1, dtype=float)
    gausfunc = lambda x: (1.0/(math.sqrt(2.0*math.pi)*fsigma))*math.exp(-(pow(x,2) / (2.0*pow(fsigma,2))))
    vfunc = np.vectorize(gausfunc)
    return vfunc(gaus)
 
# perform a gaussian filter on a 2-dimensional array
# input: sigma for standard deviation
# output: a 2-d gaussian filter with standard deviation sigma
def gauss2d(sigma):
    #compute the 2-d filter by convolving tow 1-d filters
    return signal.convolve2d(gauss1d(sigma)[np.newaxis], np.transpose(gauss1d(sigma)[np.newaxis]))


# convolve an image array with a guassian filter with standard deviation sigma
# input: a 2-d image array
# input: sigma for standard deviation
# output: a filtered image array
def gaussconvolve2d(array,sigma):
    filter = gauss2d(sigma)
    return signal.convolve2d(array,filter,'same')

def testConvolve2d():
    img_path = "dog.jpg"
    im = Image.open(img_path)
    im_array = im.convert('L')
    mod_im_array = gaussconvolve2d(im_array,3)
    gauss_im = Image.fromarray(mod_im_array)
    #show original
    im.show()
    #show filter
    gauss_im.show()
    #save filtered image
    gauss_im = gauss_im.convert('RGB')
    gauss_im.save("gaussianBandW.png", "PNG")

# runs a filter over a color image
# input: a 2-d color image
# input: sigma for standard deviation
# output: a filtered color image array
def gaussconvolveColor2d(im,sigma):
    red, green, blue = im.split()

    red_array = np.asarray(red)
    green_array = np.asarray(green)
    blue_array = np.asarray(blue)

    rgbArray = np.zeros((red_array.shape[0],red_array.shape[1],3),'uint8')
  
    new_red_array = gaussconvolve2d(red_array,sigma)
    new_green_array = gaussconvolve2d(green,sigma)
    new_blue_array = gaussconvolve2d(blue_array,sigma)

    rgbArray[..., 0] = new_red_array
    rgbArray[..., 1] = new_green_array
    rgbArray[..., 2] = new_blue_array   
    return rgbArray

# Display an image that has been filtered with a gaussian filter
# input: a 2-d color image path
# input: sigma for standard deviation
def colourGaussianDisplay(img_path, sigma):
    im = Image.open(img_path)
    rgbArray = gaussconvolveColor2d(im,sigma)
    rgb_im = Image.fromarray(rgbArray,'RGB')

    rgb_im.show()
    rgb_im.save('colorGaussian.jpeg','JPEG')
    
# Compute high frequency version of given image
# input: a 2-d color image
# output: an image array that is the high frequency of the original
def highFreq(im):
    rgb_im = im.convert('RGB')
    red, green, blue = im.split()

    red_array = np.asarray(red)
    green_array = np.asarray(green)
    blue_array = np.asarray(blue)

    rgb_array = np.zeros((red_array.shape[0],red_array.shape[1],3),'uint8')
  
    rgb_array[..., 0] = red_array
    rgb_array[..., 1] = green_array
    rgb_array[..., 2] = blue_array

    #get a low frequency version of the original
    filtered_array = gaussconvolveColor2d(im,3)
    #substract the low from the original to get the high
    highlow = np.zeros((red_array.shape[0],red_array.shape[1],3),'uint8')
    for x in range(0,3):
        highlow[...,x] = rgb_array[...,x] - filtered_array[...,x] + 128
    return highlow

# Display a high frequency example
def testHighFreq():
    img_path = "0a_cat.bmp"
    im = Image.open(img_path)
    im_arr = highFreq(im)
    mod_im = Image.fromarray(im_arr,'RGB')
    mod_im.show()
    mod_im.save("highFreq.jpeg", "JPEG")
    
# Given two color images, add them together
# input: two image arrays
# output: the combo image array
def addTwoColourImages(im_arr1, im_arr2):
    clipped_im_arr1 = np.clip(im_arr1, 0 , 255)
    clipped_im_arr2 = np.clip(im_arr2, 0 ,255)
    combo_array = np.zeros((im_arr1.shape[0],im_arr1.shape[1],3),'uint8')
    for x in range(0,3):
        combo_array[...,x] = clipped_im_arr1[...,x]/2.0 + clipped_im_arr2[...,x]/2.0
    
    return combo_array

# Display an example adding two images together 
def testAddTwoColourImages():
    img1_path = "dog.jpg"
    img2_path = "0a_cat.bmp"
    im1 = Image.open(img1_path)
    im2 = Image.open(img2_path)
    im_arr1 = gaussconvolveColor2d(im1, 3)
    im_arr2 = highFreq(im2)
    added_im_arr = addTwoColourImages(im_arr1, im_arr2)
    added_im = Image.fromarray(added_im_arr)
    added_im.show()
    added_im.save("comboImage.jpeg", "JPEG")

