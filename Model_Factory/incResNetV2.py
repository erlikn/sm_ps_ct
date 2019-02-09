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

import Model_Factory.optimizer_params as optimizer_params
import Model_Factory.IncResnetV2.inception_resnet_v2 as inception_resnet_v2
import Model_Factory.loss_base as loss_base


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
    
    ##### calls inceptionResnet and returns logits
    with tf.contrib.slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=0.0)):#self._weight_decay
        if kwargs.get('phase') == 'train': 
            # Forces is_training to False to disable batch norm update.
            with tf.contrib.slim.arg_scope([tf.contrib.slim.batch_norm],
                              is_training=True):#self._train_batch_norm
                with tf.variable_scope('InceptionResnetV2',
                                       reuse=None) as scope: # self._reuse_weights
                    fireOut, _ = inception_resnet_v2.inception_resnet_v2(
                                    images,
                                    num_classes=kwargs.get('networkOutputSize'),
                                    scope=scope, 
                                    is_training=True)
                    return fireOut
        else:
            # Forces is_training to False to disable batch norm update.
            with tf.contrib.slim.arg_scope([tf.contrib.slim.batch_norm],
                              is_training=False):
                with tf.variable_scope('InceptionResnetV2',
                                       reuse=self._reuse_weights) as scope:
                    fireOut, _ = inception_resnet_v2.inception_resnet_v2(
                                    images,
                                    num_classes=kwargs.get('networkOutputSize'),
                                    scope=scope,
                                    is_training=False)
                    return fireOut


########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

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
    return loss_base.loss(pred, target, **kwargs)



########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

def train(loss, globalStep, **kwargs):
    if kwargs.get('optimizer') == 'MomentumOptimizer':
        optimizerParams = optimizer_params.get_momentum_optimizer_params(globalStep, **kwargs)
    if kwargs.get('optimizer') == 'AdamOptimizer':
        optimizerParams = optimizer_params.get_adam_optimizer_params(globalStep, **kwargs)
    if kwargs.get('optimizer') == 'GradientDescentOptimizer':
        optimizerParams = optimizer_params.get_gradient_descent_optimizer_params(globalStep, **kwargs)
    if kwargs.get('optimizer') == 'AdaGrad':
        optimizerParams = optimizer_params.get_adagrad_optimizer_params(globalStep, **kwargs)

    # Generate moving averages of all losses and associated summaries.
    #lossAveragesOp = loss_base.add_loss_summaries(loss, kwargs.get('activeBatchSize', None))
    tf.add_to_collection('losses', loss)
    losses = tf.get_collection('losses')
    for l in losses:
        tf.summary.scalar('loss_'+l.op.name, l)


    # Compute gradients.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batchnorm
    #tvars = tf.trainable_variables()
    #with tf.control_dependencies(update_ops):
    #    if kwargs.get('optimizer') == 'AdamOptimizer':
    #        optim = tf.train.AdamOptimizer(learning_rate=optimizerParams['learningRate'], epsilon=optimizerParams['epsilon'])
    #    if kwargs.get('optimizer') == 'MomentumOptimizer':
    #        optim = tf.train.MomentumOptimizer(learning_rate=optimizerParams['learningRate'], momentum=optimizerParams['momentum'])
    #    if kwargs.get('optimizer') == 'GradientDescentOptimizer':
    #        optim = tf.train.GradientDescentOptimizer(learning_rate=optimizerParams['learningRate'])
    #    if kwargs.get('optimizer') == 'AdaGrad':
    #        optim = tf.train.AdamOptimizer(learning_rate=optimizerParams['learningRate'])
    learningRate = optimizer_params._get_learning_rate_piecewise_shifted(globalStep, **kwargs)
    optim = tf.train.GradientDescentOptimizer(learningRate)
    print(loss)
    opTrain = tf.contrib.slim.learning.create_train_op(loss, optimizer = optim)

#        grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), kwargs.get('clipNorm'))
#
#    # Apply gradients.
#    #applyGradientOp = opt.apply_gradients(grads, global_step=globalStep)
#    #train_op = opt.apply_gradients(gradsNvars, global_step=globalStep)
#    opApplyGradients = optim.apply_gradients(zip(grads, tvars), global_step=globalStep)

        
#    # Add histograms for trainable variables.
#    for var in tf.trainable_variables():    
#        tf.summary.histogram(var.op.name, var)
#    
#    # Add histograms for gradients.
#    for grad, var in zip(grads, tvars):
#        if grad is not None:
#            tf.summary.histogram(var.op.name + '/gradients', grad)
#    
#    with tf.control_dependencies([opApplyGradients]):
#        opTrain = tf.no_op(name='train')
    return opTrain

def test(loss, globalStep, **kwargs):
    return model_base.test(loss, globalStep, **kwargs)
