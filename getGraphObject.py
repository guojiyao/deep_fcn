import numpy as np
import matplotlib.pyplot as plt
import cv2
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


def one_hot_normalize(rgb):
        # return one hot vector of mask
        b=rgb[:,:,0]
        label = b/255.0
        #label = np_utils.to_categorical(label.flatten().astype(int), 2)
        #label = label.reshape(rgb.shape[0],rgb.shape[1],2)

        return label

def creatGraphObject():
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        x_shape = 400
        y_shape = 400

        for i in range(7000):
                img = cv2.imread("/mnt/data/jiyao/masks/masks_img"+str(i+1)+".tif")
                img = cv2.resize(img,(400,400))
                img = one_hot_normalize(img)
		img = np.stack([img])
                #img = np.transpose(img,[2,0,1])
                y_train.append(img)

        for i in range(7000):
                img = cv2.imread("/mnt/data/jiyao/tiles/tiles_img"+str(i+1)+".tif")
                img = cv2.resize(img,(x_shape,y_shape))
                img = normalized(img)
		img = np.stack([img])
                #img = np.transpose(img,[2,0,1])
                x_train.append(img)
	'''
        for i in range(5000,6000):
                img = cv2.imread("/mnt/data/jiyao/masks/masks_img"+str(i+1)+".tif")
                img = cv2.resize(img,(x_shape,y_shape))
                img = one_hot_normalize(img)
		img = np.stack([img])
                #img = np.transpose(img,[2,0,1])
                y_test.append(img)

        for i in range(5000,6000):
                img = cv2.imread("/mnt/data/jiyao/tiles/tiles_img"+str(i+1)+".tif")
                img = cv2.resize(img,(x_shape,y_shape))
                img = normalized(img)
		img = np.stack([img])
                #img = np.transpose(img,[2,0,1])
                x_test.append(img)
	'''
        return x_train, y_train, x_test, y_test
