# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
import os
import json
import collections
import math
import random
from shutil import copy
import numpy as np
import csv
import tensorflow as tf

from joblib import Parallel, delayed
import multiprocessing

def output(filename, npFilename, npTargetP, npTargetT, **kwargs):
    """
    """
    dictOut = {}
    for i in range(kwargs.get('activeBatchSize')):
        print(filename)
        print(npFilename[i])
        print(npTargetP[i])
        print(npTargetT[i])
        print()
        dictOut[str(npFilename[i])] = {'targetP': npTargetP[i], 'targetT': npTargetT[i]}
    _write_json_file(kwargs.get('outputDir')+filename+'.json', dictOut)
    return

def _write_json_file(filepath, datafile):
    with open(filepath, 'w') as outFile:
        json.dump(datafile, outFile)

def _read_json_file(filepath):
    with open(filepath) as dataFile:
        return json.load(open(dataFile))