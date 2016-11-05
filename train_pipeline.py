import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from keras.utils import np_utils
import logging
import tensorflow as tf
import sys
import fcn8_vgg
import logging
from loss import softmaxoutput_loss
from imageread import imageread, labelread

def getfalseP(f_p):
    positive = 1
    negative = 1
    f_positive = 0
    f_negative = 0

    for i in f_p:
        if i[0] == 0:
            positive += 1
            if i[1]<0.5: 
                f_positive+=1
        if i[1] == 0:
            negative += 1
            if i[0]<0.5:
                f_negative+=1
    return float(f_positive)/float(positive), float(f_negative)/float(negative)



logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

with tf.Session() as sess:

    # build fcn net
    images = tf.placeholder(tf.float32, [None, 400, 400, 3])
    labels = tf.placeholder(tf.int8, [None, 400, 400, 2])
    learning_rate = 1e-5
    batch_size = 1
    epoch_size = 7000

    vgg_fcn = fcn8_vgg.FCN8VGG()

    with tf.name_scope("content_vgg"):
        vgg_fcn.build(images, debug=True, train=True, num_classes=2,random_init_fc8=True)

    logging.warning("Score weights are initialized random.")
    logging.info("Start training.")

    # compute loss
    logits = vgg_fcn.pred_up
    softmax_loss,false_p = softmaxoutput_loss(logits, labels, 2)
    correct_pred = tf.equal(tf.argmax(logits,3), tf.argmax(labels,3))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()
    sess.run(init)

    #run training
    step = 1
    for epoch in range(50):
        for i in range(epoch_size):

            x = imageread(i)
            y = labelread(i)
  
            sess.run(optimizer, feed_dict={images : x, labels: y})
            if(step % 10 == 0):
                total_loss, accuracy,f_p = sess.run([loss, acc, false_p], feed_dict={images : x, labels: y})
                p,n =getfalseP(f_p)
                print( "Epoch[%d], "%(epoch) + "Iter " + str(step*batch_size) + \
                      ", Minibatch Loss= " + "{:.6f}".format(total_loss) + ", Acc = " + "{:.6f}".format(accuracy) + ", false_positive="+ "{:.6f}".format(n)+ ", false_negative="+ "{:.6f}".format(p))
            if(step % 1000 == 0):
                saver.save(sess, ('data/tf_model/%s-crowd.ckpt')%(str(epoch)),global_step=step)
            step = step + 1 

