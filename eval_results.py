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
    with open('Model_Settings/'+jsonToRead) as data_file:
        return json.load(open(filepath))

def main(modelName, phase):
    # import json_maker, update json files and read requested json file
    import Model_Settings.json_maker as json_maker
    if not json_maker.recompile_json_files(modelName):
        return
    jsonToRead = modelName+'.json'
    print("Reading %s" % jsonToRead)
    with open('Model_Settings/'+jsonToRead) as dataFile:
        modelParams = json.load(dataFile)
    
    return

if __name__ == '__main__':
    if (len(sys.argv)<3):
        raise Exception("'Enter 'model name' and 'phase (train/test)'")
    modelName = sys.argv[1]
    phase = sys.argv[2]
    main(modelName, phase)