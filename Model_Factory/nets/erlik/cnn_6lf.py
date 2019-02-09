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

from nets.erlik import model_base_2out as model_base

USE_FP_16 = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def get_modelparams():
  dataLocal = {
        # Data Parameters
        'numTrainDatasetExamples' : 23640,
        'numTestDatasetExamples' : 131,
        #'trainDataDir' : baseTrainDataDir,
        #'testDataDir' : baseTestDataDir,
        #'trainLogDir' : trainLogDirBase+'',
        #'testLogDir' : testLogDirBase+'',
        #'outputTrainDir' : trainLogDirBase+'/target/',
        #'outputTestDir' : testLogDirBase+'/target/',
        'pretrainedModelCheckpointPath' : '',
        # Image Parameters
        'pngRows' : 256,
        'pngCols' : 352,
        'pngChannels' : 1, # All PCL files should have same cols
        # Model Parameters
        'modelName' : '',
        'modelShape' : [64, 64, 64, 64, 128, 128, 128, 128, 1024],
        'batchNorm' : True,
        'weightNorm' : False,
        'optimizer' : 'MomentumOptimizer', # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        'momentum' : 0.9,
        'initialLearningRate' : 0.01,
        'learningRateDecayFactor' : 0.01,
        'numEpochsPerDecay' : 10000.0,
        'epsilon' : 0.1,
        'dropOutKeepRate' : 0.5,
        'clipNorm' : 1.0,
        'lossFunction' : 'L2',
        # Train Parameters
        'trainBatchSize' : 16,
        'testBatchSize' : 16,
        'outputSize' : 6, # 6 labels
        'trainMaxSteps' : 90000,
        'testMaxSteps' : 1,
        'usefp16' : False,
        'logDevicePlacement' : False,
        'classification' : False,
        }
  dataLocal['testMaxSteps'] = int(np.ceil(dataLocal['numTestDatasetExamples']/dataLocal['testBatchSize']))
  dataLocal['classificationModel'] = True
  dataLocal['modelName'] = 'cnn_6lf'
  dataLocal['dropOutKeepRate'] = 0.5
  dataLocal['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
  dataLocal['momentum'] = 0.9
  dataLocal['initialLearningRate'] = 0.0005
  dataLocal['learningRateDecayFactor'] = 0.1
  dataLocal['epsilon'] = 0.1
  dataLocal['modelShape'] = [64, 128, 256, 512, 256]
  dataLocal['numTrainDatasetExamples'] = 21020
  dataLocal['numTestDatasetExamples'] = 131
  dataLocal['logicalOutputSize'] = 6
  dataLocal['outputSize']=6
  dataLocal['networkOutputSize'] = dataLocal['logicalOutputSize']
  dataLocal['lossFunction'] = "clsf_smce_l2reg"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
  ######## No resizing - images are resized after parsing inside data_input.py
  #dataLocal['pngRows'] = 256
  #dataLocal['pngCols'] = 352
  dataLocal['pngChannels'] = 3
  ## runs
  dataLocal['trainMaxSteps'] = 20010
  #dataLocal['numEpochsPerDecay'] = float(dataLocal['trainMaxSteps']/3)
  #dataLocal['testMaxSteps'] = int(dataLocal['numTestDatasetExamples']/dataLocal['testBatchSize'])+1
  dataLocal['numValiDatasetExamples'] = 1024
  #dataLocal['valiSteps'] = int(dataLocal['numValiDatasetExamples']/dataLocal['trainBatchSize'])+1
  #dataLocal['trainLogDir'] = trainLogDirBase + runName
  #dataLocal['testLogDir'] = testLogDirBase + runName
  dataLocal['trainDataDir'] = '../Data/cold_wb/train_tfrecs_rgm_1/'
  dataLocal['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_rgm_1/'
  dataLocal['testDataDir'] = '../Data/cold_wb/test_tfrecs_rgm_1/'
  #dataLocal['trainOutputDir'] = dataLocal['trainLogDir']+'/target/'
  #dataLocal['testOutputDir'] = dataLocal['testLogDir']+'/target/'
  dataLocal['batchNorm'] = True
  dataLocal['weightNorm'] = False
  dataLocal['phase'] = 'train'
  return dataLocal

def get_features(images):
  return inference_l2reg(images, get_modelparams())


def inference_l2reg(images, kwargs): #batchSize=None, phase='train', outLayer=[13,13], existingParams=[]
    modelShape = kwargs.get('modelShape')
    wd = None #0.0002

#    batchSize = kwargs.get('activeBatchSize', None)

    ############# CONV1 3x3 conv, 2 input dims, 2 parallel modules, 64 output dims (filters)
    fireOut1, prevExpandDim, l2reg1 = model_base.conv_fire_module_l2regul('conv1', images, kwargs.get('pngChannels'),
                                                                  {'cnn3x3': modelShape[0]},
                                                                  wd, stride=[1,2,2,1], **kwargs)
    #fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='maxpool1')
    ############## CONV2
    fireOut2, prevExpandDim, l2reg2 = model_base.conv_fire_module_l2regul('conv2', fireOut1, prevExpandDim,
                                                                  {'cnn3x3': modelShape[1]},
                                                                  wd, stride=[1,2,2,1], **kwargs)
    #fireOut1 = tf.nn.max_pool(fireOut1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME', name='maxpool2')
    ############## CONV3
    fireOut3, prevExpandDim, l2reg3 = model_base.conv_fire_module_l2regul('conv3', fireOut2, prevExpandDim,
                                                                  {'cnn3x3': modelShape[2]},
                                                                  wd, stride=[1,4,4,1], **kwargs)
    fireOut2 = tf.nn.max_pool(fireOut2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='maxpool3')
    ############## CONV4
    fireOut4, prevExpandDim, l2reg4 = model_base.conv_fire_module_l2regul('conv4', fireOut3, prevExpandDim,
                                                                  {'cnn3x3': modelShape[3]},
                                                                  wd, stride=[1,2,2,1], **kwargs)
    #fireOut3 = tf.nn.max_pool(fireOut3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')
    # CONCAT
    #fireOut1 = tf.concat([fireOut1, fireOut2], axis=3)

#    prevExpandDim = int(fireOut1.get_shape()[3])
#    ###### DROPOUT after CONV8
#    with tf.name_scope("drop"):
#        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
#        fireOut1 = tf.nn.dropout(fireOut1, keepProb, name="dropout")
#    ############# FC1 layer with 1024 outputs
    namelist, fireOutlist, prevExpandDim, l2reg5 = model_base.conv_fire_inception_module_l2reg('convFC', fireOut4, prevExpandDim,
                                                       {'cnnFC': modelShape[4]},
                                                       wd, **kwargs)
#    # [batchsize, 1, 1, prevExpandDim]
#    fireOut1 = tf.reshape(fireOut1, [batchSize, prevExpandDim])
#    # calc batch norm FC1
#    #if kwargs.get('batchNorm'):
#    #    fireOut1 = model_base.batch_norm('batchnorm9', fireOut1, dtype, kwargs.get('phase'))
#    ############# FC1 layer with 1024 outputs
#    fireOut1, prevExpandDim, l2reg6 = model_base.fc_fire_module_l2regul('fc2', fireOut1, prevExpandDim,
#                                                       {'fc': modelShape[4]/2},
#                                                       wd, **kwargs)
#    # calc batch norm FC1
#    #if kwargs.get('batchNorm'):
#    #    fireOut1 = model_base.batch_norm('batchnorm10', fireOut1, dtype, kwargs.get('phase'))
#    ############# FC2 layer with 8 outputs
#    fireOut1, prevExpandDim, l2reg7 = model_base.fc_regression_module_l2regul('fc3', fireOut1, prevExpandDim,
#                                                             {'fc': kwargs.get('networkOutputSize')},
#                                                             wd, **kwargs)
#    l2reg = (l2reg1+l2reg2+l2reg3+l2reg4+l2reg5+l2reg6+l2reg7)/7
#    l2reg = (l2reg1+l2reg2+l2reg3+l2reg4+l2reg5+l2reg7)/6
#    l2reg = (l2reg1+l2reg2+l2reg3+l2reg5+l2reg6)/5
#    l2reg = (l2reg1+l2reg2+l2reg5+l2reg6)/4
    fireOut23 = tf.concat([fireOut2, fireOut3], axis=3)
    #return {#'conv23': fireOut23,
    return {'conv4' : fireOut4, namelist[0]: fireOutlist[0]}, namelist[1]: fireOutlist[1], namelist[2]: fireOutlist[2]}#[fireOut1, fireOut2]#, l2reg

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

def loss_l2reg(pred, target, l2reg, **kwargs): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    Returns:
      Loss tensor of type float.
    """
    return model_base.loss_l2reg(pred, target, l2reg, **kwargs)

def train(loss, globalStep, **kwargs):
    return model_base.train(loss, globalStep, **kwargs)

def test(loss, globalStep, **kwargs):
    return model_base.test(loss, globalStep, **kwargs)
