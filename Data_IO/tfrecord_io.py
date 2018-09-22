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
        'label': _float_nparray( numPassLabel.tolist()),  
    """
    pngRows = 256
    pngCols = 352
    pngCnls = kwargs.get('pngChannels') # 16bit 2channel
    labelSize = kwargs.get('logicalOutputSize')
    featureMap = {
        'temp_v': tf.FixedLenFeature([pngRows*pngCols*pngCnls], dtype=tf.float32),
        'filename': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([labelSize], dtype=tf.float32)
        }
    features = tf.parse_single_example(exampleSerialized, featureMap)
    filename = features['filename']
    pngTemp = _decode_float_image(features['temp_v'],
                                pngRows,
                                pngCols,
                                pngCnls)
    target = features['label']
    return filename, pngTemp, target

def write_tfrecords_shard(pngDatalist, filenamelist, numPassLabellist, writeFolder, shardname):
    ######## No resizing - images are resized after parsing inside data_input.py
    writer = tf.python_io.TFRecordWriter(writeFolder+'/'+shardname+'.tfrecords')
    for i in range(len(pngDatalist)):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'temp_v': _float_nparray(pngDatalist[i]),
            'filename': _bytes_feature(filenamelist[i].encode()),
            'label': _float_nparray(numPassLabellist[i])
            }))
        writer.write(tf_example.SerializeToString())
    writer.close()


