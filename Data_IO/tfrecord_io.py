# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from shutil import copy
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# to get HAB and pOrig
def _float_nparray(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _decode_byte_string(filename):
    """Decode and preprocess one filename.
    Args:
      filename: Binary string Tensor
    Returns:
      String Tensor containing the image in float32
    """
    tfname = tf.decode_raw(filename, tf.uint8)
    return tfname

def _decode_byte_image(image, height, width, depth=1):
    """Decode and preprocess one image for evaluation or training.
    Args:
      imageBuffer: Binary string Tensor
      Height, Widath, Channels <----- GLOBAL VARIABLES ARE USED DUE TO SET_SHAPE REQUIREMENTS
    Returns:
      3-D float Tensor containing the image in float32
    """
    image = tf.decode_raw(image, tf.bytes)
    image = tf.reshape(image, [height, width, depth])
    image.set_shape([height, width, depth])
    return image

def _decode_float_image(image, height, width, depth=1):
    """Decode and preprocess one image for evaluation or training.
    Args:
      imageBuffer: Binary string Tensor
      Height, Widath, Channels <----- GLOBAL VARIABLES ARE USED DUE TO SET_SHAPE REQUIREMENTS
    Returns:
      3-D float Tensor containing the image in float32
    """
    image = tf.reshape(image, [height, width, depth])
    image.set_shape([height, width, depth])
    return image

def parse_example_proto(exampleSerialized, **kwargs):
    """
        'temp_v': _float_nparray(pngData),
        'filename': _bytes_feature(str.encode(filenames[i])),
        'label6d': _float_nparray( numPassLabel.tolist()),  
    """
    pngRows = 480
    pngCols = 640
    labelSize = 6
    featureMap = {
        'temp_v': tf.FixedLenFeature([pngRows*pngCols], dtype=tf.float32),
        'filename': tf.FixedLenFeature([], dtype=tf.int64),
        'label6d': tf.FixedLenFeature([labelSize], dtype=tf.float32)
        }
    features = tf.parse_single_example(exampleSerialized, featureMap)
    filename = features['filename']
    pngTemp = _decode_float_image(features['temp_v'],
                                pngRows,
                                pngCols)
    target = features['label6d']
    return filename, pngTemp, target

def write_tfrecords(pngData, filename, numPassLabel, writeFolder):

    writer = tf.python_io.TFRecordWriter(writeFolder+'/'+str(filename)+'.tfrecords')
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'temp_v': _float_nparray(pngData),
        'filename': _int64_feature(filename),
        'label6d': _float_nparray(numPassLabel)
        }))
    writer.write(tf_example.SerializeToString())
    writer.close()