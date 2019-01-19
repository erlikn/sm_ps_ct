import tensorflow as tf
import sys
import os
import json
from PIL import Image
import cv2
import numpy as np
import time

import multiprocessing

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


def convertToEightBit(array):#, name):
    blank_image1 = np.zeros((array.shape[0],array.shape[1],3), np.uint8)   #empty image to write values to (3 channels, 8 bits each)
    high = (array/256).astype(np.int8)
    low = (array%256)
    blank_image1[:,:,0]=high
    blank_image1[:,:,1]=low
    blank_image1 = blank_image1[...,::-1]
    return blank_image1#, blank_image1

def write_tfrecord(pngFolder, filenames, writeFolder, i):
    ######## No resizing - images are resized after parsing inside data_input.py
    pngData = cv2.imread(pngFolder+'/'+filenames[i], -1)
    data = convertToEightBit(pngData)
    print(writeFolder+"/"+filenames[i])
    cv2.imwrite(writeFolder+"/"+filenames[i], data)
    return

def create_tfrecords(pngFolder, filenames, writeFolder):
    jsonData = json.load(open(sys.argv[1] + "/filenames_test.json"))

    print("Starting datawrite")
    print('Progress = 0 %')
    for j in range(len(filenames)):
        write_tfrecord(pngFolder, filenames, writeFolder, j)
    print('Progress = 100 %')
    print('Done')

    return

def _set_folder(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def main(_): 
    print('Argument List:', str(sys.argv))
    testFilenames = _read_json_file(sys.argv[1]+'/filenames_test.json')

    print("Writing test iamges...")
    _set_folder(sys.argv[1]+"/test_png_rg0_1")
    create_tfrecords(sys.argv[1] + "/pngVisual", testFilenames, sys.argv[1]+"/test_png_rg0_1")


if __name__ == '__main__':
    tf.app.run()
