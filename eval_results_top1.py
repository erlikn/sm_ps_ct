from datetime import datetime
import os.path
import time
import json
import importlib
from os import listdir                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
from os.path import isfile, join
print(os.getcwd())
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import sys

def _read_json_file(filepath):
    with open(filepath) as data_file:
        return json.load(open(filepath))

def evaluate(resDict):
    print('resDict len = ', len(resDict))
    dictKeys = resDict.keys()
    pos1 = 0
    neg1 = 0
    for k in dictKeys:
        t = np.array(resDict[k]['targetT'])
        tidx = np.argmax(t)
        p = np.array(resDict[k]['targetP'])
        pidx = np.argmax(p)
        if tidx == pidx:
            pos1 += 1
        else:
            neg1 += 1
    print("Accuracy top 1 = ", 100*pos1/(pos1+neg1))            
    print(pos1+neg1)            
    return

def _get_resultDict(jsonPath):
    filenames = listdir(jsonPath)
    resDict = {}
    for filename in filenames:
        if 'json' in filename:
            resDict.update(_read_json_file(jsonPath+filename))
    return resDict

def main(modelName, phase):
    # import json_maker, update json files and read requested json file
    import Model_Settings.json_maker as json_maker
    if not json_maker.recompile_json_files(modelName):
        return
    jsonToRead = modelName+'.json'
    print("Reading %s" % jsonToRead)
    with open('Model_Settings/'+jsonToRead) as dataFile:
        modelParams = json.load(dataFile)
    
    if phase == 'train':
        jsonPath = modelParams['trainOutputDir']
    elif phase == 'test':
        jsonPath = modelParams['testOutputDir']
    else:
        print('Please enter proper phase')
        return
    
    resultsDict = _get_resultDict(jsonPath)
    evaluate(resultsDict)
    return

if __name__ == '__main__':
    if (len(sys.argv)<3):
        raise Exception("'Enter 'model name' and 'phase (train/test)'")
    modelName = sys.argv[1]
    phase = sys.argv[2]
    main(modelName, phase)