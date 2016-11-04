"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def iou_loss(preds, t, b, l, r):
    with tf.name_scope('loss'):
        t = t 
        b = b 
        l = l 
        r = r 
        preds = tf.reshape(preds,(720,720,4))
        #preds = tf.reshape(preds,(-1,1))
        zeros = tf.zeros((720,720))

        pred_t = preds[:,:,0]
        pred_b = preds[:,:,1]
        pred_l = preds[:,:,2]
        pred_r = preds[:,:,3]

        list_box = [t,b,l,r]
        filt = tf.ones((720,720))
        for bound in list_box:
            filter_zero = tf.not_equal(bound, zeros)
            filt = tf.mul(filt, tf.cast(filter_zero, tf.float32))

        pred_t = tf.mul(preds[:,:,0], filt)
        pred_b = tf.mul(preds[:,:,1], filt)
        pred_l = tf.mul(preds[:,:,2], filt)
        pred_r = tf.mul(preds[:,:,3], filt)
        t = tf.mul(t, filt)
        b = tf.mul(b, filt)
        l = tf.mul(l, filt)
        r = tf.mul(r, filt)


        x_p = tf.mul(tf.add(t , b), tf.add(l , r))
        x = tf.mul(tf.add(pred_t , pred_b), tf.add(pred_l , pred_r))
        I_h = tf.add(tf.minimum(t, pred_t) , tf.minimum(b, pred_b))
        I_w = tf.add(tf.minimum(l, pred_l) , tf.minimum(r, pred_r))

        epsilon = tf.constant(value=1e-6)
        I = tf.mul(I_h, I_w) + epsilon
        U = x + x_p - I + epsilon * 2
        IoU = tf.truediv(I,U)

        iouloss = tf.truediv(tf.reduce_sum(-tf.log(IoU)), tf.reduce_sum(filt))
        tf.add_to_collection('losses', iouloss)
    return  iouloss

def reg_loss_x(preds, labels, weight = 1.0):
    with tf.name_scope('loss'):
        preds = tf.reshape(preds, (-1,1))
        labels = tf.to_float(tf.reshape(labels, (-1, 1)))

        #mask the back_ground
        ignore = tf.mul(tf.ones((720*720, 1)) , -10000.0)
        filted_back = tf.not_equal(labels, ignore)

        #filtered_labels = tf.truediv(tf.mul(labels, tf.cast(filted_back, tf.float32)) , 720.0)
        filtered_labels = tf.truediv( tf.mul(labels, tf.cast(filted_back, tf.float32)), 720.0)
        filtered_preds = tf.mul(preds, tf.cast(filted_back, tf.float32))

        # regress_mean = -tf.reduce_mean(filtered_preds - filtered_labels, name='regress_mean')
        regress_mean = tf.reduce_mean(tf.pow(filtered_preds - filtered_labels,2 )) 
        tf.add_to_collection('losses', regress_mean)
    return regress_mean


def softmaxoutput_loss(logits, labels, num_classes, head=None, weight = 1.0):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        logits = logits + epsilon
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits)

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

    return cross_entropy_mean
