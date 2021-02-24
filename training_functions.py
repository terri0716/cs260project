from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image  
import PIL
import matplotlib
import cv2


# This contains training functions!
class ColorClass:
    def __init__ (self, mean, cov, prior):
        self.mean = mean
        self.cov = cov
        self.prior = prior
        
# Separate train_data into pixel-level
def computeGaussian(train_data, class_num):
    # Input: training data
    # Output: a list of color class objects 
    
    train_num, pixel_num = train_data.shape[:2]
    pixel_data = train_data.reshape(-1,4)
    pixels = pixel_data[:,0:3]
    labels = pixel_data[:,3]

    class_objs = []

    for i in range(class_num):
        # Locate pixels of class i
        mask = (labels==i+1)
        class_pixels = pixels[mask,:]
        
        # Compute gaussian mean parameters
        mean = np.mean(class_pixels, axis=0).astype('float32')
        covar = np.cov(class_pixels.T, rowvar=True).astype('float32')
        prior = class_pixels.shape[0]/(pixel_num*train_num)
        class_objs.append(ColorClass(mean,covar,prior))
        
    return class_objs
        
        

        
        
        


        