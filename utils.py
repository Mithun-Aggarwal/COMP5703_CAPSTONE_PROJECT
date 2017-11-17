# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:35:32 2017

@author: magg5201
For COMP 5703 - Capstone project 
Unsupervised learning with Gan for LF data

This is the Util module
helps in itial extraction of LF image & plotiing of final result 
"""

#imported libraries
import numpy as np
import scipy.misc
import tensorflow as tf
import glob
import cv2
import os
from scipy.ndimage import rotate

FLAGS = tf.app.flags.FLAGS

#Function used for expoting samples at checkpoint to disk
def grid_plot(images, size, path):
    h, w = FLAGS.output_image_height, FLAGS.output_image_width
    image_grid = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        image_grid[int(j * h):int(j * h + h), int(i * w):int(i * w + w), :] = image
    scipy.misc.imsave(path, image_grid)

#Function used for Converting 1 LF image to subviwes
#This needs to be Rethreaded
def ont_to_64(path,img_path,dim,cut_x,cut_y,test_path):
   for filename in glob.glob(path+'/*.png'):
       print(str(path))
       img_2D = cv2.imread(filename)
       img_2D = img_2D[:,:,0:3]       
       h = int(np.floor(img_2D.shape[0] / dim[0] ))
       w = int(np.floor(img_2D.shape[1] / dim[1] ))
       c = int(np.floor(img_2D.shape[2]))
       print("h : "+str(h)+" w : "+str(w)+" c : "+str(c))
       fullInput = np.zeros((dim[0],dim[1],h,w,c))
       
       for i in range(dim[0]):
           for j in range(dim[1]):
               img = img_2D[i::dim[0],j::dim[1],  0:3]
               fullInput[i,j,:,:,:] = img[:h, :w]
       fullInput = fullInput[(0+cut_x):(dim[0]-cut_x),\
                             (0+ cut_y):(dim[1] - cut_y),:,:,:]
       fullInput = np.reshape(fullInput, [((dim[0]-(2*cut_x))*(dim[1]-(2*cut_y))),h,w,3])
       save_img = np.zeros((h,w,c))
       total = 0
       for img_no in range(fullInput.shape[0]):
           save_img = fullInput[img_no,:,:,:]
           save_img = np.reshape(save_img, [h,w,c])
           total += 1
           if (FLAGS.agumentation_rotate == 'True'):
               if (img_no != 30):
                   agum_img = rotate(save_img, 5*img_no, reshape=False)
                   scipy.misc.imsave(img_path+str(os.path.basename(filename)+"_"+str(img_no))+".png",agum_img)
               elif (img_no == 30):
                   scipy.misc.imsave(test_path+str(os.path.basename(filename)+"_"+str(img_no))+".png",save_img)
           elif (FLAGS.agumentation_rotate == 'False'):
               if (img_no != 30):
                   scipy.misc.imsave(img_path+str(os.path.basename(filename)+"_"+str(img_no))+".png",save_img)
               elif (img_no == 30):
                   scipy.misc.imsave(test_path+str(os.path.basename(filename)+"_"+str(img_no))+".png",save_img)