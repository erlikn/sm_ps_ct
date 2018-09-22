import tensorflow as tf
import sys
import os
import json
from PIL import Image
import cv2
import numpy as np
import time

import multiprocessing

import Data_IO.tfrecord_io as tfrecord_io

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

shardNumber = 0
samplesinshard = 512
listData = list()
listFlname = list()
listNumPassLabel = list()

np.set_printoptions(precision=2, suppress=True)

def _read_json_file(filename):
    return json.load(open(filename))

def write_tfrecord(pngFolder, filenames, jsonData, jsonFileName, writeFolder, i):
    ######## No resizing - images are resized after parsing inside data_input.py
    data = cv2.imread(pngFolder+'/'+filenames[i], -1)
    print(data.shape)
    #print(data.shape)
    #print(np.sum(np.sum(data, 0), 0))
    #cv2.imshow('mix', data)
    #cv2.waitKey(0)
    

    # Label preparation
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
               # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
               # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    try:
        jidx = jsonFileName.index(filenames[i])
    except:
        print(filenames[i])
        print('Doesnt have a corresponding json entry')
        return
    numPassengers = 0
    for j in range(0,len(jsonData['frames'][jidx]['annotations'])):
        if(jsonData['frames'][jidx]['annotations'][j]['label'] == 'Head'):
            numPassengers += 1
    if numPassengers==0:
        numPassLabel = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif numPassengers==1:
        numPassLabel = np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)
    elif numPassengers==2:
        numPassLabel = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)
    elif numPassengers==3:
        numPassLabel = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
    elif numPassengers==4:
        numPassLabel = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
    elif numPassengers==5:
        numPassLabel = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    else:
      print('Error.... more than 5 passengers!!! -->', numPassengers)
    
    global shardNumber
    global listData
    global listFlname
    global listNumPassLabel

    listData.append(np.reshape(data, -1))
    listFlname.append(filenames[i])
    listNumPassLabel.append(numPassLabel)

    if (len(listData) == samplesinshard) or (i == len(filenames)-1):
        tfrecord_io.write_tfrecords_shard(listData, listFlname, listNumPassLabel, writeFolder, str(shardNumber))
        shardNumber+=1
        listData.clear()
        listFlname.clear()
        listNumPassLabel.clear()
        #if ((10*i)%len(filenames) < 1):
        print('Progress = ', 100*i/len(filenames), '%  -  Writing to shard ', shardNumber)
    return

def create_tfrecords(pngFolder, filenames, writeFolder):
    # TODO(user): Populate the following variables from your example.
    height = 256 # Image height
    width = 352 # Image width
    encoded_image_data = None # Encoded image bytes
    image_format = 'png' # b'jpeg' or b'png``'

    jsonData = json.load(open(sys.argv[1] + "/combined.json"))
    jsonFileName = []
    for i in range(len(jsonData['frames'])):
        jsonFileName.append(jsonData['frames'][i]['file'])

    print("Starting datawrite")
    #startTime = time.time()
    print('Progress = 0 %')
    for j in range(len(filenames)):
        write_tfrecord(pngFolder, filenames, jsonData, jsonFileName, writeFolder, j)
    print('Progress = 100 %')
    print('Done')
    #print(time.time()-startTime)

    return

def _set_folder(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def main(_): 
    print('Argument List:', str(sys.argv))
    trainFilenames = _read_json_file(sys.argv[1]+'/filenames_train.json')
    from random import shuffle
    shuffle(trainFilenames)
    testFilenames = _read_json_file(sys.argv[1]+'/filenames_test.json')

    print("Writing train records...")
    _set_folder(sys.argv[1]+"/train_tfrecs_352")
    create_tfrecords(sys.argv[1] + "/trainpng352", trainFilenames, sys.argv[1]+"/train_tfrecs_352")
    print("Writing test records...")
    _set_folder(sys.argv[1]+"/test_tfrecs_352")
    create_tfrecords(sys.argv[1] + "/testpng352", testFilenames, sys.argv[1]+"/test_tfrecs_352")


if __name__ == '__main__':
    tf.app.run()
