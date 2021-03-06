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
import shutil

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

PHASE = 'test'

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
	#params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])
	modelParams['existingParams'] = None

	if modelParams['phase'] == 'train':
		modelParams['activeBatchSize'] = modelParams['trainBatchSize']
		modelParams['maxSteps'] = modelParams['trainMaxSteps']
		modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
		modelParams['dataDir'] = modelParams['trainDataDir']
		modelParams['logDir'] = modelParams['trainLogDir']
		modelParams['outputDir'] = modelParams['trainOutputDir']

	if modelParams['phase'] == 'test':
		modelParams['activeBatchSize'] = modelParams['testBatchSize']
		modelParams['maxSteps'] = modelParams['testMaxSteps']
		modelParams['numExamples'] = modelParams['numTestDatasetExamples']
		modelParams['dataDir'] = modelParams['testDataDir']
		modelParams['logDir'] = modelParams['testLogDir']
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

	_setupLogging(os.path.join(modelParams['logDir'], "genlog"))

	with tf.Graph().as_default():
		# track the number of train calls (basically number of batches processed)
		globalStep = tf.get_variable('globalStep',
									 [],
									 initializer=tf.constant_initializer(0),
									 trainable=False)

		# Get images inputs for model_cnn.
		if modelParams['phase'] == 'v':
			filename, pngTemp, targetT = data_input.inputs_vali(**modelParams)
		else:
			filename, pngTemp, targetT = data_input.inputs(**modelParams)
		print('Input        ready')

		# Build a Graph that computes the HAB predictions from the
		# inference model
		targetP = model_cnn.inference(pngTemp, **modelParams)
		#print(targetP.get_shape())
		# loss model
		#if modelParams.get('classificationModel'):
		#	print('Classification model...')
		#	# loss on last tuple
		#	loss = model_cnn.loss(targetP, targetT, **modelParams)
		#else:
		#	print('Regression model...')
		#	# loss on last tuple
		#	loss = model_cnn.loss(targetP, targetT, **modelParams)
			
		##############################
		print('Training     ready')
		# Create a saver.
		saver = tf.train.Saver(tf.global_variables())
		print('Saver        ready')

		# Build the summary operation based on the TF collection of Summaries.
		summaryOp = tf.summary.merge_all()
		print('MergeSummary ready')
		# Build an initialization operation to run below.
		#init = tf.initialize_all_variables()
		init = tf.global_variables_initializer()

		#opCheck = tf.add_check_numerics_ops()
		# Start running operations on the Graph.
		config = tf.ConfigProto(log_device_placement=modelParams['logDevicePlacement'])
		config.gpu_options.allow_growth = True
		config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
		sess = tf.Session(config=config)
		print('Session      ready')
		
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
		sess.run(init)

		# restore a saver.
		print('Loading Ex-Model with epoch number %d ...', epochNumber)
		saver.restore(sess, (modelParams['trainLogDir']+'/model.ckpt-'+str(epochNumber)))
		#saver.restore(sess, (modelParams['trainLogDir']+'_30k/model.ckpt-29000'))
		print('Ex-Model     loaded')

		tf.train.write_graph(sess.graph.as_graph_def(), '.' , modelParams['trainLogDir']+'/model.pbtxt', as_text=True)
		
		# Start the queue runners.
		tf.train.start_queue_runners(sess=sess)
		print('QueueRunner  started')
		
		summaryWriter = tf.summary.FileWriter(modelParams['logDir'], sess.graph)
#TEST###        summaryValiWriter = tf.summary.FileWriter(modelParams['logDir']+'_test', sess.graph)
		total_parameters = 0
		for variable in tf.trainable_variables():
			# shape is an array of tf.Dimension
			shape = variable.get_shape()
			#print(shape)
			#print(len(shape))
			variable_parameters = 1
			for dim in shape:
				#print(dim)
				variable_parameters *= dim.value
				#print(variable_parameters)
			total_parameters += variable_parameters
		print('-----total parameters-------- ', total_parameters)


		if True:
			# if True: freeze graph
			tf.train.write_graph(sess.graph.as_graph_def(), '.' , modelParams['trainLogDir']+'_v/model.pbtxt', as_text=True)
			# Output nodes
			output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
			# Freeze the graph
			frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
			# Save the frozen graph
			with open(modelParams['trainLogDir']+'_v/model.pb', 'wb') as f:
				f.write(frozen_graph_def.SerializeToString())
				
		print('Training     started')
		durationSum = 0
		durationSumAll = 0
		prevLoss = 99999
		prevValiSumLoss = 99999
		prevaccur = 0
		prevLossStep = 0
		prevStep = 21000
		l = list()
#TEST###        prevTestSumLoss = 99999
		prevStep = int(modelParams['maxSteps']/2)
		for step in xrange(0, modelParams['maxSteps']):
			startTime = time.time()
			lossValue = sess.run([targetP])
			#print(lossValue, l2regValue)
			duration = time.time() - startTime
			if step != 0:
				l.append(duration)
			durationSum += duration
			
			#assert not np.isnan(lossValue), 'Model diverged with loss = NaN'

			print(duration, step, modelParams['maxSteps'])
		print(np.array(l).mean())

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
	global PHASE
	modelParams['phase'] = PHASE
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

	if epochNumber == 0:
		#if input("(Overwrite WARNING) Did you change logs directory? (y) ") != "y":
		#    print("Please consider changing logs directory in order to avoid overwrite!")
		#    return
		if tf.gfile.Exists(modelParams['trainLogDir']):
			tf.gfile.DeleteRecursively(modelParams['trainLogDir'])
		tf.gfile.MakeDirs(modelParams['trainLogDir'])
	train(modelParams, epochNumber)


if __name__ == '__main__':
	# looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
	# list and invokes that modules main function (defined above)
	#    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
	#    - run can be called with the specific main and/or arguments
	tf.app.run()
