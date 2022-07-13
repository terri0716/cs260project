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


def equalization(data):
    sample_num, pixel_num = data.shape[:2]
    equalized_data = data.copy()
    
    labels = data[:,:,3]
    images = data[:,:,0:3]
    
    for i in range(sample_num):
        image = images[i].reshape(1200,900,3).astype('uint8')
        
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        image_y = image_ycrcb[:,:,0]
        
        # adaptive histogram
        #equ = cv2.equalizeHist(image_y)
        
        # clahe
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        equ = clahe.apply(image_y)
        
        equ_ycrcb = image_ycrcb.copy()
        equ_ycrcb[:,:,0] = equ
        final = cv2.cvtColor(equ_ycrcb, cv2.COLOR_YCrCb2RGB)
        final_vector = final.reshape(pixel_num,3).astype('int')
        
        equalized_data[i,:,0:3] = final_vector
    
    return equalized_data
        
        
     
        
        
        
    
        
        

        
        
        


        