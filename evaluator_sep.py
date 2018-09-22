#import write_outputs

import sys
import json
import shutil
import time
from pathlib import Path
import os
import collections

def write_json_file(filename, datafile):
	datafile = collections.OrderedDict(sorted(datafile.items()))
	with open(filename, 'w') as outFile:
		json.dump(datafile, outFile, indent=0)

#############################################

#def main(argv=None):
#	argv = sys.argv
#	if (len(argv)<3):
#		print("Enter 'model name'")
#	modelName = argv[1]
#	testStep = argv[2]
#	resDict = {}
#	fileJSON = Path(modelName+'.json')
#	if fileJSON.is_file():
#		with open(modelName+'.json') as data_file:
#			resDict = json.load(data_file)
#	epochSteps = 1000
#	import Model_Settings.json_maker as json_maker
#	jsonToRead = modelName+'.json'
#	print("Reading %s" % jsonToRead)
#	with open('Model_Settings/'+jsonToRead) as data_file:
#		modelParams = json.load(data_file)
#	
#	argv.append(0)
#	copy = False
#	bestStep = 0
#	bestAcc = 0
#	bestAccb = 0
#	for evalStep in range(int(testStep), int(testStep)+1):#modelParams['trainMaxSteps'], epochSteps):
#		print('Waiting for 	   ', modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep))
#		while True:
#			fileCheckpoint = Path(modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.data-00000-of-00001')
#			if fileCheckpoint.is_file():
#				print('		copying...')
#				shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.data-00000-of-00001', modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep)+'.data-00000-of-00001' )
#				shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.index', modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep)+'.index' )
#				shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.meta', modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep)+'.meta' )
#				break
#			time.sleep(60)
#
#main()



import eval_results
import eval_results_b
import write_outputs_test
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(argv=None):
	argv = sys.argv
	if (len(argv)<3):
		print("Enter 'model name'")
	modelName = argv[1]
	testStep = argv[2]
	resDict = {}
	fileJSON = Path(modelName+'.json')
	if fileJSON.is_file():
		with open(modelName+'.json') as data_file:
			resDict = json.load(data_file)
	epochSteps = 1000
	import Model_Settings.json_maker as json_maker
	jsonToRead = modelName+'.json'
	print("Reading %s" % jsonToRead)
	with open('Model_Settings/'+jsonToRead) as data_file:
		modelParams = json.load(data_file)
	
	argv.append(0)
	copy = False
	bestStep = 0
	bestAcc = 0
	bestAccb = 0
	for evalStep in range(int(testStep), int(testStep)+1):#modelParams['trainMaxSteps'], epochSteps):
		print('Waiting for 	   ', modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep))
		while True:
			fileCheckpoint = Path(modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep)+'.data-00000-of-00001')
			if fileCheckpoint.is_file():
				break
			time.sleep(60)
			

		argv[2] = evalStep
		print(argv)
		write_outputs_test.main(argv)
		acc = eval_results.main(argv[1], 'test')
		accb = eval_results_b.main(argv[1], 'test')
		print('----')
		print('----')
		print('---- 	curr	', evalStep, ' - ', acc, ' - ', accb )
		print('---- 	best 	', bestStep, ' - ', bestAcc, ' - ', bestAccb )
		accdict = {str(evalStep):[acc, accb]}
		resDict.update(accdict)
		print('---- 	writing 	', modelName+'.json')		
		write_json_file(modelName+'.json', resDict)
		print('----')
		print('----')
		if acc > bestAcc:
			bestAcc = acc
			bestAccb = accb
			bestStep = evalStep
			#print('     Saving model')
			#shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.data-00000-of-00001', modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep)+'.data-00000-of-00001' )
			#shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.index', modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep)+'.index' )
			#shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.meta', modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep)+'.meta' )
	while True:
		print('Waiting to Read   ', modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep+1000)+'.data-00000-of-00001')
		fileCheckpoint = Path(modelParams['trainLogDir']+'_validation/model.ckpt-'+str(evalStep+1000)+'.data-00000-of-00001')
		if fileCheckpoint.is_file():
			break
		time.sleep(60)
	
if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()
