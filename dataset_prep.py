import tensorflow as tf
import sys
import os
import json
import PIL
import cv2
import numpy as np

from joblib import Parallel, delayed
import multiprocessing

import Data_IO.tfrecord_io as tfrecord_io

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def write_tfrecord(pngFolder, filenames, jsonData, jsonFileName, writeFolder, i):
    pngFile = PIL.Image.open(pngFolder+'/'+filenames[i])
    pngData = pngFile.getdata()
    pngData = np.array(pngData, dtype=np.float32)

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

    tfrecord_io.write_tfrecords(pngData.tolist(), 100000+i, numPassLabel.tolist(), writeFolder)
    if ((10*i)%len(filenames) < 1):
        print('Progress = ', 100*i/len(filenames), '%')


def create_tfrecords(pngFolder, filenames, writeFolder):
    # TODO(user): Populate the following variables from your example.
    height = 480 # Image height
    width = 640 # Image width
    encoded_image_data = None # Encoded image bytes
    image_format = 'png' # b'jpeg' or b'png``'

    jsonData = json.load(open(sys.argv[1] + "/combined.json"))
    jsonFileName = []
    for i in range(len(jsonData['frames'])):
        jsonFileName.append(jsonData['frames'][i]['file'])

    print("Starting datawrite")
    num_cores = multiprocessing.cpu_count() - 2
    #for j in range(0, len(jsonData['frames'])):
    #  write_tfrecord(pngFolder, filenames, jsonData, jsonFileName, writeFolder, j)
    Parallel(n_jobs=num_cores)(delayed(write_tfrecord)(pngFolder, filenames, jsonData, jsonFileName, writeFolder, j) for j in range(5443,len(jsonData['frames'])))
    print('Done')

    return

def main(_): 
    print('Argument List:', str(sys.argv))
    filenames = os.listdir(sys.argv[1] + "/png")
    for file in filenames:
        if not 'png' in file:
            filenames.remove(file)
    create_tfrecords(sys.argv[1] + "/png", filenames, sys.argv[1]+"/tfrecords")


if __name__ == '__main__':
    tf.app.run()
