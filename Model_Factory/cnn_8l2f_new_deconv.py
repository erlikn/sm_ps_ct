# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the calusa_heatmap network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

import Model_Factory.model_base_new as model_base

USE_FP_16 = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def inference(images, **kwargs): #batchSize=None, phase='train', outLayer=[13,13], existingParams=[]
    modelShape = kwargs.get('modelShape')
    wd = None #0.0002
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    batchSize = kwargs.get('activeBatchSize', None)

    ############# CONV1 3x3 conv, 2 input dims, 2 parallel modules, 64 output dims (filters)
    fireOut1, prevExpandDim = model_base.conv_fire_module('conv1', images, kwargs.get('pngChannels'),
                                                                  {'cnn3x3': modelShape[0]},
                                                                  wd, stride=[1,2,2,1], **kwargs)
    ############# CONV2
    fireOut1, prevExpandDim = model_base.conv_fire_module('conv2', fireOut1, prevExpandDim,
                                                                  {'cnn3x3': modelShape[1]},
                                                                  wd, **kwargs)
    ############# CONV3
    fireOut2, prevExpandDim = model_base.conv_fire_module('conv3', fireOut1, prevExpandDim,
                                                                  {'cnn3x3': modelShape[2]},
                                                                  wd, stride=[1,2,2,1], **kwargs)
    fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME', name='maxpool1')
    ############# CONV4
    fireOut2, prevExpandDim = model_base.conv_fire_module('conv4', fireOut2, prevExpandDim,
                                                                  {'cnn3x3': modelShape[3]},
                                                                  wd, **kwargs)
    ############# CONV5
    fireOut3, prevExpandDim = model_base.conv_fire_module('conv5', fireOut2, prevExpandDim,
                                                                  {'cnn3x3': modelShape[4]},
                                                                  wd, stride=[1,2,2,1], **kwargs)
    fireOut2 = tf.nn.max_pool(fireOut2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='maxpool2')
    ############# CONV6
    fireOut3, prevExpandDim = model_base.conv_fire_module('conv6', fireOut3, prevExpandDim,
                                                                  {'cnn3x3': modelShape[5]},
                                                                  wd, **kwargs)
    ############# CONV7
    fireOut4, prevExpandDim = model_base.conv_fire_module('conv7', fireOut3, prevExpandDim,
                                                                  {'cnn3x3': modelShape[6]},
                                                                  wd, stride=[1,2,2,1], **kwargs)
    fireOut3 = tf.nn.max_pool(fireOut3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool2')
    ############# CONV8
    fireOut4, prevExpandDim = model_base.conv_fire_module('conv8', fireOut4, prevExpandDim,
                                                                  {'cnn3x3': modelShape[7]},
                                                                  wd, **kwargs)
    # CONCAT
    fireOut1 = tf.concat([fireOut1, fireOut2, fireOut3, fireOut4], axis=3)
    prevExpandDim = int(fireOut1.get_shape()[3])
    ###### DROPOUT after CONV8
    with tf.name_scope("drop"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        fireOut1 = tf.nn.dropout(fireOut1, keepProb, name="dropout")
    ###### Prepare for fully connected layers
    #fireOut = tf.reshape(fireOut, [batchSize, -1])
    #prevExpandDim = int(fireOut1.get_shape()[1])
    ############## FC1 layer with 1024 outputs
    #fireOut, prevExpandDim = model_base.fc_fire_module('fc1', fireOut, prevExpandDim,
    #                                                   {'fc': modelShape[8]},
    #                                                   wd, **kwargs)
    ############# FC1 layer with 1024 outputs
    fireOut1, prevExpandDim = model_base.conv_fire_inception_module('convFC', fireOut1, prevExpandDim,
                                                       {'cnnFC': modelShape[8]},
                                                       wd, **kwargs)
    # [batchsize, 1, 1, prevExpandDim]
    fireOut1 = tf.reshape(fireOut1, [batchSize, prevExpandDim])
    # calc batch norm FC1
    if kwargs.get('batchNorm'):
        fireOut1 = model_base.batch_norm('batchnorm9', fireOut1, dtype, kwargs.get('phase'))
    ############# FC1 layer with 1024 outputs
    #fireOut1, prevExpandDim = model_base.fc_fire_module('fc2', fireOut1, prevExpandDim,
    #                                                   {'fc': modelShape[9]},
    #                                                   wd, **kwargs)
    ## calc batch norm FC1
    #if kwargs.get('batchNorm'):
    #    fireOut1 = model_base.batch_norm('batchnorm10', fireOut1, dtype, kwargs.get('phase'))
    ############# FC2 layer with 8 outputs
    fireOut1, prevExpandDim = model_base.fc_regression_module('fc3', fireOut1, prevExpandDim,
                                                             {'fc': kwargs.get('networkOutputSize')},
                                                             wd, **kwargs)
    return fireOut1

def loss(pred, target, **kwargs): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    Returns:
      Loss tensor of type float.
    """
    return model_base.loss(pred, target, **kwargs)

def train(loss, globalStep, **kwargs):
    return model_base.train(loss, globalStep, **kwargs)

def test(loss, globalStep, **kwargs):
    return model_base.test(loss, globalStep, **kwargs)
