

# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import json

#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

import Data_IO.tfrecord_io as tfrecord_io


# 32 batches (64 smaples per batch) = 2048 samples in a shard
TRAIN_SHARD_SIZE = 2*2
TEST_SHARD_SIZE = 2*2
#190 shard files with (2048 samples per shard)
CHNAGE_TO_TOTAL_FILE_NUMBER = 20400
NUMBER_OF_SHARDS = (CHNAGE_TO_TOTAL_FILE_NUMBER//TRAIN_SHARD_SIZE)+1 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('trainShardIdent', 'train',
                           """How to identify training shards. Name must start with this token""")

#tf.app.flags.DEFINE_string('val_shard_ident', 'val',
#                           """How to identify validation shards.
#                              Name must start with this token""")

# dataset sample size is 25000
tf.app.flags.DEFINE_string('testShardIdent', 'test',
                           """How to identify testing shards. Name must start with this token""")

# dataset sample size is 388915
tf.app.flags.DEFINE_integer('numberOfShards', NUMBER_OF_SHARDS, 
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('trainShardSize', TRAIN_SHARD_SIZE,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('testShardSize', TEST_SHARD_SIZE,
                            'Number of shards in training TFRecord files.')

#tf.app.flags.DEFINE_integer('val_shard_size', 8*25,    # 200 records/shard
#                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('numPreprocessThreads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")

tf.app.flags.DEFINE_integer('numReaders', 2,
                            """Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --inputQueueMemoryFactor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('inputQueueMemoryFactor', 2,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")

def validate_for_nan(tensorT):
    # Input:
    #   Tensor
    # Output:
    #   0 if contains a NaN or Inf value
    #   1 if it is valid
    tensorMean = tf.reduce_mean(tensorT)
    validity = tf.select(tf.is_nan(tensorMean), 0, 1) * tf.select(tf.is_inf(tensorMean), 0, 1)
    return validity

def image_preprocessing(image, **kwargs):
    """Decode and preprocess one image for evaluation or training.
    Args:
      imageBuffer: 3D tf.float32 Tensor (height, width, channels)
    Returns:
      3-D float Tensor containing an appropriately scaled image
    """
    ############################ if each dim has a different range then mean it per dim
    meanChannels, stdChannels = tf.nn.moments(image, axes=[1, 2]) # b*r*c*d => d times meanChannels
    # SUBTRACT MEAN
    meanChannels = tf.reshape(meanChannels, [kwargs.get('activeBatchSize'), 1,1,-1]) # prepare for channel based subtraction
    imagenorm = tf.subtract(image, meanChannels)
    # DIVIDE BY STANDARD DEVIATION
    stdChannels = tf.reshape(stdChannels, [kwargs.get('activeBatchSize'), 1,1,-1]) # prepare for channel based scalar division
    stdChannels = tf.cond(tf.reduce_all(tf.not_equal(stdChannels,tf.zeros_like(stdChannels))),
                          lambda: stdChannels,
                          lambda: tf.ones_like(stdChannels))
    imagenorm = tf.div(imagenorm, stdChannels)
    # No need to scale to [-1,1] -> it is 0 mean and small values. That is ala we need.
    # SCALE TO [-1,1]
    #maxChannels = tf.reduce_max(imagenorm, [0,1])
    #minChannels = tf.reduce_min(imagenorm, [0,1])
    #maxminDif = tf.subtract(maxChannels, minChannels)
    #maxminSum = tf.add(maxChannels, minChannels)
    #maxminDif = tf.reshape(maxminDif, [1,1,-1])
    #maxminSum = tf.reshape(maxminSum, [1,1,-1])
    #coef = tf.constant(2.0, shape=[1,1,2])
    #maxminDif = tf.cond(tf.reduce_all(tf.not_equal(maxminDif,tf.zeros_like(maxminDif))),
    #                                  lambda: maxminDif, 
    #                                  lambda: tf.ones_like(maxminDif))
    #imagenorm = tf.div(tf.subtract(tf.multiply(coef,imagenorm), maxminSum), maxminDif) # ((2*x)-(max+min))/(max-min)
    return imagenorm

def fetch_inputs(dataDir, numPreprocessThreads=None, numReaders=1, **kwargs):
    """Construct input for DeepHomography using the Reader ops.
    Args:
      dataDir: Path to the DeepHomography data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, imageDepthRows, imageDepthCols, 2] size.
      target: transformation matrix. 3D tensor of [batch_size, 1, 12=targetRows*targetCols] size.
      pclA: Point Clouds. 3D tensor of [batch_size, pclRows, pclCols]
      pclB: Point Clouds. 3D tensor of [batch_size, pclRows, pclCols]
      tfRecfileID: 3 ints [seqID, frame i, frame i+1]
    """    
    with tf.name_scope('batch_processing'):
        # get dataset filenames
        filenames = glob.glob(os.path.join(dataDir, "*.tfrecords"))
        print(dataDir)
        # read parameters
        ph = kwargs.get('phase')
        if filenames is None or len(filenames) == 0:
            raise ValueError("No filenames found for stage: %s" % ph)
        '''
        for f in filenames_orig:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)
        '''
        # create input queue
        if ph == 'train':
            # Create a queue that produces the filenames to read.
            filenameQueue = tf.train.string_input_producer(filenames,
                                                           shuffle=True,
                                                           capacity=8)
        else:
            filenameQueue = tf.train.string_input_producer(filenames,
                                                           shuffle=False,
                                                           capacity=8)
        # set number of preprocessing threads
        if numPreprocessThreads is None:
            numPreprocessThreads = FLAGS.numPreprocessThreads
        if numPreprocessThreads % 4:
            raise ValueError('Please make numPreprocessThreads a multiple '
                             'of 4 (%d % 4 != 0).', numPreprocessThreads)
        # set number of readers
        if numReaders is None:
            numReaders = FLAGS.numReaders
        if numReaders < 1:
            raise ValueError('Please make numReaders at least 1')
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 sample (two 128x128 greyscale images) uses 128*128*2*4 bytes ~ 0.14MB
        # miniBatchSize is 64
        # 1 shard is about 32 batches = 2048 samples (287 MB/shard)
        # The default inputQueueMemoryFactor is 8 implying a shuffling queue
        # size of 16*287 MB ~ 4.6 GB
        if kwargs.get('phase') == 'train':
            # calculate number of examples per shard.
            examplesPerShard = FLAGS.trainShardSize
            minQueueExamples = examplesPerShard * FLAGS.inputQueueMemoryFactor
            # create example queue place holder
            examplesQueue = tf.RandomShuffleQueue(
                capacity=minQueueExamples + 3 * kwargs.get('activeBatchSize'),
                min_after_dequeue=minQueueExamples,
                dtypes=[tf.string])
        else:
            # calculate number of examples per shard.
            examplesPerShard = FLAGS.testShardSize
            minQueueExamples = examplesPerShard * FLAGS.inputQueueMemoryFactor
            # create example queue place holder
            examplesQueue = tf.RandomShuffleQueue(
                capacity=minQueueExamples + 3 * kwargs.get('activeBatchSize'),
                min_after_dequeue=minQueueExamples,
                dtypes=[tf.string])
        # read examples, put in the queue, and generate serialized examples
        if numReaders > 1:
            enqueue_ops = []
            for _ in range(numReaders):
                reader = tf.TFRecordReader()
                _, value = reader.read(filenameQueue)
                enqueue_ops.append(examplesQueue.enqueue([value]))

            # ?
            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examplesQueue, enqueue_ops))
            # generate serialized example
            exampleSerialized = examplesQueue.dequeue()

        else:
            reader = tf.TFRecordReader()
            # generate serialized example
            _, exampleSerialized = reader.read(filenameQueue) 
        # Read data from queue
        sampleData = []
        for _ in range(numPreprocessThreads):
            # Parse a serialized Example proto to extract the image and metadata.
            filename, pngTemp, target = tfrecord_io.parse_example_proto(exampleSerialized, **kwargs)
            sampleData.append([filename, pngTemp, target])
            
        batchFilename, batchPngTemp, batchTarget = tf.train.batch_join(sampleData,
                                                                    batch_size=kwargs.get('activeBatchSize'),
                                                                    capacity=2*numPreprocessThreads*kwargs.get('activeBatchSize'))

        # Resize to the desired size
        batchPngTemp = tf.image.resize_images(batchPngTemp, tf.constant([kwargs.get('pngRows'),kwargs.get('pngCols')], dtype=tf.int32))

        batchPngTemp = tf.cast(batchPngTemp, tf.float32)
        # Display the training images in the visualizer.
        for i in range(kwargs.get('pngChannels')):
            tf.summary.image('tempImg_'+str(i)+'_', tf.split(batchPngTemp, 2, 3)[0])
        
        #batchPngTemp = image_preprocessing(batchPngTemp, **kwargs)
        return batchFilename, batchPngTemp, batchTarget

def inputs(mode='train or test', **kwargs):
    """Construct input for DeepHomography_CNN evaluation using the Reader ops.
    
    Args:

    Returns:
      batchImage: Images. 4D tensor of [batch_size, 128, 512, 2] size.
    Raises:
      ValueError: If no dataDir
    """
    dataDir = kwargs.get('dataDir')
    if mode == 'vali':
        dataDir = kwargs.get('valiDataDir')
    
    if not dataDir:
        raise ValueError('Please supply a dataDir')

    with tf.device('/cpu:0'):
        batchFilename, batchPngTemp, batchTarget = fetch_inputs(dataDir, **kwargs)
        
        if kwargs.get('usefp16'):
            batchPngTemp = tf.cast(batchPngTemp, tf.float16)
            batchTarget = tf.cast(batchTarget, tf.float16)

    return batchFilename, batchPngTemp, batchTarget
