from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image  
import PIL
import matplotlib
import cv2


# This contains testing functions!

def getPixelEstimates(test_data, class_num, class_objs):
    # Input: testing data, class objects list
    # Output: List of label estimates of all images
    
    test_num, pixel_num = test_data.shape[:2]
    estimates_allimages = []
    twopi3 = np.power(2*np.pi,3)
    
    for i in range(test_num):
        # Extract one image data 
        image = test_data[i][:,:3]
        label = test_data[i][:,3]
        posteriors_list = []
        
        for j in range(class_num):
            # Get class properties, for class j
            cov = class_objs[j].cov
            prior = class_objs[j].prior
            
            # Compuete posterior probability, for class j
            temp = np.matmul(image, np.linalg.inv(cov))
            temp2 = np.einsum('ij,ij->i', temp, image)
            likelihood = -1/2*temp2 - 1/2*np.log(twopi3 * np.linalg.det(cov))
            posterior = likelihood + np.log(prior)
            posteriors_list.append(posterior)
        
        # Gather posteroir probability of all classes 
        posteriors = np.stack(posteriors_list).T
        # Compute pixel label estimate 
        estimate = np.argmax(posteriors, axis=1) + 1
        estimates_allimages.append(estimate)
        
    return estimates_allimages


def makeImage(array):
    # takes in array of pixels (pixelnum, 3)
    # returns matrix-image
    return array.reshape([1200,900,3]).astype("uint8")
    # plt.imsave('name.extention', makeImage(array))

def showGreenImages_getBinaryImages(test_data, test_index, estimates_allimages):
    
    test_num, pixel_num = test_data.shape[:2]
    binary_images = []
    
    for i in range( test_num):
        # Extract one image data
        image = test_data[i][:,:3]
        label = test_data[i][:,3]
        
        mask_index = (estimates_allimages[i]==1)
        # Show green images
        newimage = image.copy()
        newimage[mask_index,:] = [0,255,0]        
        plt.figure(i)
        plt.imshow(makeImage(newimage))
        plt.title('{}'.format(test_index[i]))
        
        # Return binary images
        newimage2 = np.zeros_like(image)
        newimage2[mask_index,:] = [255,255,255]
        binary_images.append(newimage2)
    
    return binary_images

def showBoundingBox(binary_images, extent_clip):
    
    test_num = len(binary_images)
    
    for i in range(test_num):
        print('image {}'.format(i))
        # Extract one image data
        img = makeImage(binary_images[i].copy())
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        image, contours, hierarchy = cv2.findContours(imgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # Areas of all contours
        areas = [cv2.contourArea(c) for c in contours]
        # Set contour: Contour that has the biggest area
        cnt = contours[np.argmax(areas)]
        
        #Test another contour-decision metohd:
        n = len(areas)
        areas_descend_ind = np.argsort(-1*np.array(areas))[:n]
        
        for j in range(n):
            #if (j>0): break
            index = areas_descend_ind[j]
            area = areas[index]
            cnt = contours[index]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxarea = cv2.contourArea(box)   
            extent = area/(boxarea+0.0001)
            print(extent)
            if extent > 0.55:
                break
            

         
        
        # Draw rotated bounding box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        boxedImage = img.copy()
        cv2.drawContours(boxedImage,[box],0,(0,255,0),10)
        
        plt.figure()
        plt.imshow(boxedImage)
        plt.title('{}'.format(extent))
                  
        
    
    
               
