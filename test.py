import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

from loss import softmaxoutput_loss
from keras.utils import np_utils
from PIL import Image

import fcn8_vgg
import utils
from geTestObject import creatTestObject

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
			stream=sys.stdout)

x_test = creatTestObject()
with tf.Session() as sess:
    images = tf.placeholder(tf.float32, [None, 400, 400, 3])
    labels = tf.placeholder(tf.int8, [None, 400, 400, 2])
    learning_rate = 1e-6
    batch_size = 1

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(images, debug=False, num_classes=2,random_init_fc8=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")
    print 'Loading the Network'
    logits = vgg_fcn.pred_up
    softmax_loss = softmaxoutput_loss(logits, labels, 2) 
    correct_pred = tf.equal(tf.argmax(logits,3), tf.argmax(labels,3))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(softmax_loss, global_step=global_step)
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()
    sess.run(init)

    saved_model = 'data/tf_model/0-crowd.ckpt-1000'
    print 'loading model..............' + saved_model
    saver.restore(sess,saved_model)
    count = 0
    for img in x_test:
        feed_dict = {images: img}
        tensors = vgg_fcn.pred_up
        up = sess.run(tensors, feed_dict=feed_dict)
        up = tf.reshape(up[0], (-1, 2))
        up = tf.nn.softmax(up)
        up = up.eval()[:,1]
        up = up.reshape((400,400))
        up = up > 0.1
        scp.misc.imsave('test_data/fcn8_'+str(count+1)+'.png', up)
        count += 1
