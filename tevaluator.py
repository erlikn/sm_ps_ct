#import write_outputs

import sys
import json
import shutil
import time
from pathlib import Path
import os
import collections
import eval_results
import eval_results_b
import twrite_outputs_test
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def write_json_file(filename, datafile):
	datafile = collections.OrderedDict(sorted(datafile.items()))
	with open(filename, 'w') as outFile:
		json.dump(datafile, outFile, indent=0)

#############################################

def main(argv=None):
	phase = 'test'
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
		#print('Waiting for 	   ', modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep))
		#while True:
		#	fileCheckpoint = Path(modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.data-00000-of-00001')
		#	if fileCheckpoint.is_file():
		#		print('		copying...')
		#		shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.data-00000-of-00001', modelParams['trainLogDir']+'_v/model.ckpt-'+str(evalStep)+'.data-00000-of-00001' )
		#		shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.index', modelParams['trainLogDir']+'_v/model.ckpt-'+str(evalStep)+'.index' )
		#		shutil.copy( modelParams['trainLogDir']+'/model.ckpt-'+str(evalStep)+'.meta', modelParams['trainLogDir']+'_v/model.ckpt-'+str(evalStep)+'.meta' )
		#		break
		#	time.sleep(30)

		###### uncomment only for testing
		fileCheckpoint = Path(modelParams['trainLogDir']+'_v/model.ckpt-'+str(evalStep)+'.data-00000-of-00001')
		print('Waiting for 	   ', modelParams['trainLogDir']+'_v/model.ckpt-'+str(evalStep))
		while True:
			if fileCheckpoint.is_file():
				break
			time.sleep(10)
		
		argv[2] = evalStep
		print("-------------------- EVALUATION STARTED -----------------")
		print(argv)
		print("-------------------- ENTER write_outputs --------------")
		twrite_outputs_test.main(argv)
		print("-------------------- exit  write_outputs ---------------")
		print("-------------------- ENTER eval_results... Calculating accuracy")
		acc = eval_results.main(argv[1], phase)
		print("-------------------- exit  eval_results... Calculating accuracy")
		print("-------------------- ENTER eval_results Binary")
		accb = eval_results_b.main(argv[1], phase)
		print("-------------------- exit  eval_results Binary")
		print('----')
		print('----')
		print('---- 	curr	', evalStep, ' - ', acc, ' - ', accb )
		print('---- 	best 	', bestStep, ' - ', bestAcc, ' - ', bestAccb )
		keyid = list(resDict.keys())
		if len(keyid)<1:
			resDict = {str(evalStep):[acc, accb]}
		else:
			maxAcc = resDict[keyid[0]][0]
			maxAccb = resDict[keyid[0]][1]
			if acc > maxAcc or (acc == maxAcc and accb > maxAccb):
				#accdict = {str(evalStep):[acc, accb]}
				#resDict.update(accdict)
				resDict = {str(evalStep):[acc, accb]}
		print('---- 	writing 	', modelName+'.json')		
		write_json_file(modelName+'.json', resDict)
		print('----')
		print('----')
		if acc > bestAcc:
			bestAcc = acc
			bestAccb = accb
			bestStep = evalStep
		print("------------- EVALUATION COMPLETED -------------")

#main()

	
if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()
