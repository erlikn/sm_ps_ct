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
                                                                  wd, **kwargs)
    ## Pooling1 2x2 wit stride 2
    #fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    #                      padding='SAME', name='maxpool1')
    # calc batch norm CONV1
    if kwargs.get('batchNorm'):
        fireOut1 = model_base.batch_norm('batchnorm1', fireOut1, dtype, kwargs.get('phase'))
    ############# CONV2
    fireOut1, prevExpandDim = model_base.conv_fire_module('conv2', fireOut1, prevExpandDim,
                                                                  {'cnn3x3': modelShape[1]},
                                                                  wd, **kwargs)
    ## Pooling1 2x2 wit stride 2
    #fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    #                      padding='SAME', name='maxpool2')
    # calc batch norm CONV2
    if kwargs.get('batchNorm'):
        fireOut1 = model_base.batch_norm('batchnorm2', fireOut1, dtype, kwargs.get('phase'))
    ############# CONV3
    fireOut1, prevExpandDim = model_base.conv_fire_module('conv3', fireOut1, prevExpandDim,
                                                                  {'cnn3x3': modelShape[2]},
                                                                  wd, **kwargs)
    # Pooling1 2x2 wit stride 2
    fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    # calc batch norm CONV3
    if kwargs.get('batchNorm'):
        fireOut2 = model_base.batch_norm('batchnorm3', fireOut1, dtype, kwargs.get('phase'))
    ############# CONV4
    fireOut2, prevExpandDim = model_base.conv_fire_module('conv4', fireOut2, prevExpandDim,
                                                                  {'cnn3x3': modelShape[3]},
                                                                  wd, **kwargs)
    ## Pooling1 2x2 wit stride 2
    #fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    #                      padding='SAME', name='maxpool2')
    #fireOut2 = tf.nn.max_pool(fireOut2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    #                      padding='SAME', name='maxpool4')
    # calc batch norm CONV4
    if kwargs.get('batchNorm'):
        fireOut2 = model_base.batch_norm('batchnorm4', fireOut2, dtype, kwargs.get('phase'))
    ############# CONV5
    fireOut2, prevExpandDim = model_base.conv_fire_module('conv5', fireOut2, prevExpandDim,
                                                                  {'cnn3x3': modelShape[4]},
                                                         wd, **kwargs)
    ## Pooling1 2x2 wit stride 2
    #fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    #                      padding='SAME', name='maxpool2')
    #fireOut2 = tf.nn.max_pool(fireOut2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    #                      padding='SAME', name='maxpool5')
    # calc batch norm CONV5
    if kwargs.get('batchNorm'):
        fireOut2 = model_base.batch_norm('batchnorm5', fireOut2, dtype, kwargs.get('phase'))
    ############# CONV6
    fireOut2, prevExpandDim = model_base.conv_fire_module('conv6', fireOut2, prevExpandDim,
                                                                  {'cnn3x3': modelShape[5]},
                                                         wd, **kwargs)
    ###### Pooling1 2x2 wit stride 2
    fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool2')
    fireOut2 = tf.nn.max_pool(fireOut2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool2')
    # calc batch norm CONV6
    if kwargs.get('batchNorm'):
        fireOut3 = model_base.batch_norm('batchnorm6', fireOut2, dtype, kwargs.get('phase'))
    ############# CONV7
    fireOut3, prevExpandDim = model_base.conv_fire_module('conv7', fireOut3, prevExpandDim,
                                                                  {'cnn3x3': modelShape[6]},
                                                         wd, **kwargs)
    # calc batch norm CONV7
    if kwargs.get('batchNorm'):
        fireOut3 = model_base.batch_norm('batchnorm7', fireOut3, dtype, kwargs.get('phase'))
    ############# CONV8
    fireOut3, prevExpandDim = model_base.conv_fire_module('conv8', fireOut3, prevExpandDim,
                                                                  {'cnn3x3': modelShape[7]},
                                                         wd, **kwargs)
    ###### Pooling1 2x2 wit stride 2
    fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    fireOut2 = tf.nn.max_pool(fireOut2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    fireOut3 = tf.nn.max_pool(fireOut3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    # calc batch norm CONV8
    if kwargs.get('batchNorm'):
        fireOut3 = model_base.batch_norm('batchnorm8', fireOut3, dtype, kwargs.get('phase'))
    # CONCAT
    fireOut = tf.concat([fireOut1, fireOut2, fireOut3], axis=3)
    prevExpandDim = int(fireOut.get_shape()[3])
    ############# CONV9
    fireOut, prevExpandDim = model_base.conv_fire_module('convFC1', fireOut, prevExpandDim,
                                                             {'cnnFC': modelShape[7]},
                                                             wd, **kwargs)
    fireOut = tf.reshape(fireOut, [batchSize, modelShape[7]])
    prevExpandDim = prevExpandDim = int(fireOut.get_shape()[1])
    ############# CONV9
    fireOut, prevExpandDim = model_base.fc_fire_module('fc0', fireOut, prevExpandDim,
                                                             {'fc': modelShape[8]},
                                                             wd, **kwargs)
    # calc batch norm CONV8
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batchnorm8', fireOut, dtype, kwargs.get('phase'))    
    ###### DROPOUT after CONV8
    with tf.name_scope("drop"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout")
    ############# FC2 layer with 8 outputs
    fireOut, prevExpandDim = model_base.fc_regression_module('fc1', fireOut, prevExpandDim,
                                                             {'fc': kwargs.get('networkOutputSize')},
                                                             wd, **kwargs)
    ###### Normalize vectors to have output [0~1] for each batch
    # fireout is [16] x [192] 
    # fireout should be [16] x [6] x [32] x [1] => now normalize for each batch and each row
    # To do so, we could rearrange everything in [16 x 6] x [32] and calculate softmax for each row and return back to original
    # kwargs.get('networkOutputSize')/kwargs.get('logicalOutputSize') = kwargs.get('classificationModel')['binSize']
    #fireOut = tf.reshape(fireOut, [kwargs.get('activeBatchSize')*kwargs.get('logicalOutputSize'), np.int32(kwargs.get('networkOutputSize')/kwargs.get#('logicalOutputSize'))])
    #fireOut.set_shape([kwargs.get('activeBatchSize')*kwargs.get('logicalOutputSize'), np.int32(kwargs.get('networkOutputSize')/kwargs.get('logicalOutputSize'))])
    #fireOut = tf.nn.softmax(fireOut)

    #### NOW CONVERT IT TO Correct format
    # kwargs.get('networkOutputSize')/kwargs.get('logicalOutputSize') = kwargs.get('classificationModel')['binSize']
    
    #fireOut = tf.reshape(fireOut, [kwargs.get('activeBatchSize'), 
    #                                   kwargs.get('logicalOutputSize'), 
    #                                   np.int32(kwargs.get('networkOutputSize')/(kwargs.get('logicalOutputSize'))), 
    #                                   1])
    #fireOut.set_shape([kwargs.get('activeBatchSize'), 
    #                       kwargs.get('logicalOutputSize'), 
    #                       np.int32(kwargs.get('networkOutputSize')/(kwargs.get('logicalOutputSize'))), 
    #                       1])

    return fireOut

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
