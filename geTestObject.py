import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import scipy.misc
import os

from keras.utils import np_utils

def normalized(rgb):
        #return rgb/255.0
        norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        norm[:,:,0]=b/255.0
        norm[:,:,1]=g/255.0
        norm[:,:,2]=r/255.0

        return norm

def creatTestObject():
        x_test = []
        x_shape = 400
        y_shape = 400
	
	for i in range(200):
                img = skimage.io.imread("data/tiles/tiles_img"+str(i+1)+".tif")
                img = skimage.transform.resize(img,(x_shape,y_shape))
                img = normalized(img)
                img = np.stack([img])
                #img = np.transpose(img,[2,0,1])
                x_test.append(img)
	return x_test
