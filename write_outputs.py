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
from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import os, os.path
import time
import logging
import json
import importlib

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# import input & output modules 
import Data_IO.data_input as data_input
import Data_IO.data_output as data_output
PHASE = 'train'

####################################################
####################################################
####################################################
####################################################
####################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('printOutStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('summaryWriteStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('modelCheckpointStep', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportStep', 250,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportOutputWrite', 250,
                            """Number of batches to run.""")
####################################################
####################################################
def _set_control_params(modelParams):
    modelParams['phase'] = PHASE
    #params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])
    modelParams['existingParams'] = None

    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']
        modelParams['outputDir'] = modelParams['trainOutputDir']
        
    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']
        modelParams['outputDir'] = modelParams['testOutputDir']

    return modelParams
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
def train(modelParams, epochNumber):
    # import corresponding model name as model_cnn, specifed at json file
    model_cnn = importlib.import_module('Model_Factory.'+modelParams['modelName'])
    
    if not os.path.exists(modelParams['dataDir']):
        raise ValueError("No such data directory %s" % modelParams['dataDir'])

    _setupLogging(os.path.join(modelParams['trainLogDir'], "genlog"))

    with tf.Graph().as_default():
        # track the number of train calls (basically number of batches processed)
        globalStep = tf.get_variable('globalStep',
                                     [],
                                     initializer=tf.constant_initializer(0),
                                     trainable=False)

        # Get images inputs for model_cnn.
        filename, pngTemp, targetT = data_input.inputs(**modelParams)
        print('Input        ready')

        # Build a Graph that computes the HAB predictions from the
        # inference model
        targetP = model_cnn.inference(pngTemp, **modelParams)
            
        ##############################
        print('Training     ready')
        # Build an initialization operation to run below.
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=modelParams['logDevicePlacement'])
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        print('Session      ready')

        sess.run(init)

        # restore a saver.
        print('Loading Ex-Model with epoch number %d ...', epochNumber)
        saver.restore(sess, (modelParams['trainLogDir']+'/model.ckpt-'+str(epochNumber)))
        print('Ex-Model     loaded')

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        print('QueueRunner  started')

        summaryWriter = tf.summary.FileWriter(modelParams['trainLogDir'], sess.graph)
        
        print('Training     started')
        durationSum = 0
        durationSumAll = 0
        for step in xrange(epochNumber, modelParams['maxSteps']):
            startTime = time.time()
            npfilename, npTargetP, npTargetT = sess.run([filename, targetP, targetT])
            duration = time.time() - startTime
            durationSum += duration

            data_output.output(str(10000+step), npfilename, npTargetP, npTargetT, **modelParams)
            
            # Print Progress Info
            if ((step % FLAGS.ProgressStepReportStep) == 0) or ((step+1) == modelParams['maxSteps']):
                print('Progress: %.2f%%, Elapsed: %.2f mins, Training Completion in: %.2f mins --- %s' %
                        (
                            (100*step)/modelParams['maxSteps'],
                            durationSum/60,
                            (((durationSum*modelParams['maxSteps'])/(step+1))/60)-(durationSum/60),
                            datetime.now()
                        )
                    )

def _setupLogging(logPath):
    # cleanup
    if os.path.isfile(logPath):
        os.remove(logPath)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logPath,
                        filemode='w')

    # also write out to the console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))

    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    logging.info("Logging setup complete to %s" % logPath)

def main(argv=None):  # pylint: disable=unused-argumDt
    if (len(argv)<3):
        print("Enter 'model name' and 'epoch number to load / 0 for new'")
        return
    modelName = argv[1]
    epochNumber = int(argv[2])
    # import json_maker, update json files and read requested json file
    import Model_Settings.json_maker as json_maker
    if not json_maker.recompile_json_files(modelName):
        return
    jsonToRead = modelName+'.json'
    print("Reading %s" % jsonToRead)
    with open('Model_Settings/'+jsonToRead) as data_file:
        modelParams = json.load(data_file)

    modelParams = _set_control_params(modelParams)

    print(modelParams['modelName'])
    print('Training steps = %.1f' % float(modelParams['trainMaxSteps']))
    print('Rounds on datase = %.1f' % float((modelParams['trainBatchSize']*modelParams['trainMaxSteps'])/modelParams['numTrainDatasetExamples']))
    print('lossFunction = ', modelParams['lossFunction'])
    print('Train Input: %s' % modelParams['trainDataDir'])
    #print('Test  Input: %s' % modelParams['testDataDir'])
    print('Train Logs Output: %s' % modelParams['trainLogDir'])
    #print('Test  Logs Output: %s' % modelParams['testLogDir'])
    print('')
    print('')

    print(modelParams['modelName'])
    print('Rounds on datase = %.1f' % float((modelParams['trainBatchSize']*modelParams['trainMaxSteps'])/modelParams['numTrainDatasetExamples']))
    print('lossFunction = ', modelParams['lossFunction'])
    print('Train Input: %s' % modelParams['trainDataDir'])
    #print('Test  Input: %s' % modelParams['testDataDir'])
    print('Train Logs Output: %s' % modelParams['trainLogDir'])
    #print('Test  Logs Output: %s' % modelParams['testLogDir'])
    print('')
    print('')

    train(modelParams, epochNumber)


if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()