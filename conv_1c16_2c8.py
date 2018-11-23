import tensorflow as tf
import sys
import os
import json
import PIL
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

#convert(): function that loops through 16 bit single channel png images in 'images_folder', and converts them to 3 channel 8 bit images (RG0) 
def convert():
    images_folder = "/home/wassimea/Desktop/SMATS/images/png/test"
    png_files = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
    for i in range(0,len(png_files)):
        print(str(png_files[i]))
        png = PIL.Image.open(images_folder + "/" + png_files[i])
        data = png.getdata()
        convertToEightBit(data,png_files[i])    #SEND THE image data and the filename to the convertToEightBit function

#convertToEightBit(): function that loops through 640x480 png data, gets two 8 bit numbers (high & low) representing the 16bit pixel value, sets the R and G values of the 8 bit image to high and low respectively, sets the third channel value to 0
def convertToEightBit(array, name):
    blank_image = np.zeros((480,640,3), np.uint8)   #empty image to write values to (3 channels, 8 bits each)
    counter1 = 1;
    counter = 0
    for y in range(480):
        for x in range(640):
            firstNumber = array[counter]    #initial 16 bit value
            high = ((firstNumber >> 8) & 0xFF)  #first 8 bit value
            low = firstNumber & 0xFF    #second 8 bit value
            blank_image[y,x] = (high,low,0)  #set channel values of current pixel 
            counter = counter + 1
    im = PIL.Image.fromarray(blank_image, 'RGB')
    x = im.getpixel((128,96))
    im.save("/home/wassimea/Desktop/SMATS/images/3c/test/" +str(name))  #save image

def main(_): 
  print('Argument List:', str(sys.argv))
  convert()

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()

