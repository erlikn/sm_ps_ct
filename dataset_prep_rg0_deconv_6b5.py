import tensorflow as tf
import sys
import os
import json
import cv2
import numpy as np
import time

import Data_IO.tfrecord_io as tfrecord_io

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

shardNumber = 0
samplesinshard = 512
listData = list()
listFlname = list()
listNumPassLabel = list()
list_heatmap = list()
list_bbox = list()

np.set_printoptions(precision=2, suppress=True)

def _read_json_file(filename):
    return json.load(open(filename))

def add_gaussian_3D(heat_img, gt):
    #gaussed_image = np.zeros((480,640,1), np.uint8)
    gaussed_image_3d = np.zeros((32, 44, 6), np.uint8)
    pass_counter=1
    for head in gt:
        gaussed_image = np.zeros((heat_img.shape[0], heat_img.shape[1]), np.uint8)
        x = head[0]
        y = head[1]
        width = head[2]
        height = head[3]
        center_x = x + (width/2)
        center_y = y + (height/2)
        if(width != 0 and height != 0):
            gx = 0.05* cv2.getGaussianKernel(width, int(width/4))
            gy = 0.05* cv2.getGaussianKernel(height, int(height/4))
            gy = np.transpose(gy)
            product = np.multiply(gx,gy)

            product = (255) * (product-np.min(product)) / (np.max(product)-np.min(product))
            w = np.max(product)
            product = product.flatten()
            count = 0
            xrange = x + width
            if xrange > heat_img.shape[1]:
                xrange = heat_img.shape[1]
            yrange = y + height
            if yrange > heat_img.shape[0]:
                yrange = heat_img.shape[0]
            for i in range(x,xrange):
                for j in range(y, yrange):
                    gaussed_image[j,i] = product[count]
                    count += 1
            t = 1
        else:
            print("WRONG IMAGE")
        gaussed_image_3d[:,:,pass_counter] = cv2.resize(gaussed_image, (44, 32))
        pass_counter+=1
    return gaussed_image_3d

def add_gaussian(heat_img, gt):
    #gaussed_image = np.zeros((480,640,1), np.uint8)
    gaussed_image = np.zeros((heat_img.shape[0], heat_img.shape[1]), np.uint8)
    pass_counter=1
    for head in gt:
        x = head[0]
        y = head[1]
        width = head[2]
        height = head[3]
        center_x = x + (width/2)
        center_y = y + (height/2)
        if(width != 0 and height != 0):
            gx = 0.05* cv2.getGaussianKernel(width, int(width/4))
            gy = 0.05* cv2.getGaussianKernel(height, int(height/4))
            gy = np.transpose(gy)
            product = np.multiply(gx,gy)

            product = (pass_counter*40) * (product-np.min(product)) / (np.max(product)-np.min(product))
            pass_counter+=1
            w = np.max(product)
            product = product.flatten()
            count = 0
            xrange = x + width
            if xrange > heat_img.shape[1]:
                xrange = heat_img.shape[1]
            yrange = y + height
            if yrange > heat_img.shape[0]:
                yrange = heat_img.shape[0]
            for i in range(x,xrange):
                for j in range(y, yrange):
                    if(gaussed_image[j,i] == 0):
                        gaussed_image[j,i] = product[count]
                    else:
                        gaussed_image[j,i] = max(gaussed_image[j,i], product[count])
                    count += 1
            t = 1
        else:
            print("WRONG IMAGE")
    gaussed_image = cv2.resize(gaussed_image, (44, 32))
    return gaussed_image

def add_padded_box(heat_img, gt):
    gaussed_image = np.zeros((heat_img.shape[0], heat_img.shape[1]), np.uint8)
    # rank heads from large to small
    new_gt=[]
    new_gt_area = []
    for head in gt:
        append_to_end = True
        width = head[2]
        height = head[3]
        for hidx in range(len(new_gt)):
            if width*height > new_gt_area[hidx]:
                new_gt.insert(hidx, head)
                new_gt_area.insert(hidx, width*height)
                append_to_end = False
        if append_to_end:#len(new_gt==0):#includes empty
            new_gt.append(head)
            new_gt_area.append(width*height)
    gt = new_gt
    for head in gt:
        x = head[0]
        y = head[1]
        width = head[2]
        height = head[3]
        center_x = x + (width/2)
        center_y = y + (height/2)
        if(width != 0 and height != 0):
            gx = cv2.getGaussianKernel(width, int(width/5))
            gy = cv2.getGaussianKernel(height, int(height/5))
            gy = np.transpose(gy)
            product = np.multiply(gx,gy)

            product = (255) * (product-np.min(product)) / (np.max(product)-np.min(product))
            w = np.max(product)
            product = product.flatten()
            count = 0
            xrange = x + width
            if xrange > heat_img.shape[1]:
                xrange = heat_img.shape[1]
            yrange = y + height
            if yrange > heat_img.shape[0]:
                yrange = heat_img.shape[0]
            for i in range(x,xrange):
                for j in range(y, yrange):
                    #gaussed_image[j,i] = product[count]
                    if(gaussed_image[j,i] == 0):
                        gaussed_image[j,i] = product[count]
                    else:
                        gaussed_image[j,i] = max(gaussed_image[j,i], product[count])
                    count += 1
            t = 1
        else:
            print("WRONG IMAGE")
    #cv2.imshow('image5', gaussed_image)
    #cv2.waitKey()
    gaussed_image = cv2.resize(gaussed_image, (44, 32))
    return gaussed_image


def get_heatmap(label, heat_img):
    filename = label['file']
    gt = []
    for j in range(0,len(label['annotations'])):
        if label['annotations'][j]['label'] == 'Head':
            x = int(label['annotations'][j]['x']/2)
            y = int(label['annotations'][j]['y']/2)
            width = int(label['annotations'][j]['width']/2)
            height = int(label['annotations'][j]['height']/2)
            if(x + width > 320):
                width = 320 - x
            if(y + height > 240):
                height = 240 - y
            gt.append([x,y,width,height])
    #heatmap = add_gaussian(heat_img, gt)
    #heatmap = add_gaussian_3D(heat_img, gt)
    heatmap = add_padded_box(heat_img, gt)
    #cv2.imshow('image', heatmap)
    #cv2.waitKey(0)
    return heatmap


def convertToEightBit(array):#, name):
    blank_image1 = np.zeros((array.shape[0],array.shape[1],3), np.uint8)   #empty image to write values to (3 channels, 8 bits each)
    high = (array/256).astype(np.int8)
    low = (array%256)
    blank_image1[:,:,0]=high
    blank_image1[:,:,1]=low
    #blank_image1 = blank_image1[...,::-1]
    return blank_image1#, blank_image1

def write_tfrecord(pngFolder, filenames, jsonData, jsonFileName, writeFolder, i):
    ######## No resizing - images are resized after parsing inside data_input.py
    #print(pngFolder+'/'+filenames[i])
    pngData = cv2.imread(pngFolder+'/'+filenames[i], -1)
    #print(pngData.shape)
    try:
        asd = pngData.shape
    except:
        print('EXCEPTION   -  ', pngFolder+'/'+filenames[i])
    
    if pngData.shape[0]!=256 or pngData.shape[1]!=352:
        print('WRONG IMAGE SIZE   -  ', pngData.shape, pngFolder+'/'+filenames[i])

    data = convertToEightBit(pngData)
   
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
    heatmap = get_heatmap(jsonData['frames'][jidx], pngData)
    bbox = np.zeros(shape=[6, 5], dtype=float)
    numPassengers = 0
    for j in range(0,len(jsonData['frames'][jidx]['annotations'])):
        if(jsonData['frames'][jidx]['annotations'][j]['label'] == 'Head'):
            if jsonData['frames'][jidx]['annotations'][j]['width'] != 0 and jsonData['frames'][jidx]['annotations'][j]['height'] != 0:
                #print(jsonData['frames'][jidx]['annotations'][j])
                numPassengers += 1
                bbox[numPassengers][0] = 1
                bbox[numPassengers][1] = jsonData['frames'][jidx]['annotations'][j]['x']/640
                bbox[numPassengers][2] = jsonData['frames'][jidx]['annotations'][j]['y']/480
                bbox[numPassengers][3] = (jsonData['frames'][jidx]['annotations'][j]['x']+jsonData['frames'][jidx]['annotations'][j]['width'])/640
                if bbox[numPassengers][3] > 1:
                    print('xmax', bbox[numPassengers][3], filenames[i])
                    bbox[numPassengers][3]=1
                bbox[numPassengers][4] = (jsonData['frames'][jidx]['annotations'][j]['y']+jsonData['frames'][jidx]['annotations'][j]['height'])/480
                if bbox[numPassengers][4] > 1:
                    print('ymax', bbox[numPassengers][4], filenames[i])
                    bbox[numPassengers][4]=1
                #print(jsonData['frames'][jidx]['annotations'][j]['x'])
                #print(jsonData['frames'][jidx]['annotations'][j]['y'])
                #print(jsonData['frames'][jidx]['annotations'][j]['width'])
                #print(jsonData['frames'][jidx]['annotations'][j]['height'])
                #print(bbox[numPassengers][1])
                #print(bbox[numPassengers][2])
                #print(bbox[numPassengers][3])
                #print(bbox[numPassengers][4])
                assert (bbox[numPassengers][1]>=0 and bbox[numPassengers][1]<=1)
                assert (bbox[numPassengers][2]>=0 and bbox[numPassengers][2]<=1)
                assert (bbox[numPassengers][3]>=0 and bbox[numPassengers][3]<=1)
                assert (bbox[numPassengers][4]>=0 and bbox[numPassengers][4]<=1)
                #bbox[numPassengers][10][7] = 0
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
    global list_heatmap
    global list_bbox

    listData.append(np.reshape(data, -1))
    listFlname.append(filenames[i])
    listNumPassLabel.append(numPassLabel)
    list_heatmap.append(np.reshape(heatmap, -1))
    list_bbox.append(np.reshape(bbox,-1))

    if (len(listData) == samplesinshard) or (i == len(filenames)-1):
        tfrecord_io.write_tfrec_heatmap_6b5(listData, listFlname, listNumPassLabel, list_heatmap, list_bbox, writeFolder, str(shardNumber))
        shardNumber+=1
        print(writeFolder)
        listData.clear()
        listFlname.clear()
        listNumPassLabel.clear()
        list_heatmap.clear()
        list_bbox.clear()
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
    #num_cores = multiprocessing.cpu_count() - 2

    #startTime = time.time()
    print('Progress = 0 %')
    filenames = filenames[:1000]
    for j in range(len(filenames)):
        write_tfrecord(pngFolder, filenames, jsonData, jsonFileName, writeFolder, j)
    #Parallel(n_jobs=num_cores)(delayed(write_tfrecord)(pngFolder, filenames, jsonData, jsonFileName, writeFolder, j) for j in range(0,len(filenames)))
    print('Progress = 100 %')
    print('Done')
    #print(time.time()-startTime)

    return

def _set_folder(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def main(_):
    id_fold = '2'
    print('Argument List:', str(sys.argv))
    sys.argv.append('../Data/cold_wb')
    trainFilenames = _read_json_file(sys.argv[1]+'/filenames_train'+id_fold+'.json')
    from random import shuffle
    shuffle(trainFilenames)
    testFilenames = _read_json_file(sys.argv[1]+'/filenames_test'+id_fold+'.json')

    print("Writing train records...")
    _set_folder(sys.argv[1]+"/train_tfrecs_rg0_f"+id_fold)
    create_tfrecords(sys.argv[1] + "/trainpngfoldK"+id_fold, trainFilenames, sys.argv[1]+"/train_tfrecs_rg0_deconv_f"+id_fold)
    print("Writing test records...")
    _set_folder(sys.argv[1]+"/test_tfrecs_rg0_f"+id_fold)
    create_tfrecords(sys.argv[1] + "/testpngfoldK"+id_fold, testFilenames, sys.argv[1]+"/test_tfrecs_rg0_deconv_f"+id_fold)


if __name__ == '__main__':
    tf.app.run()
