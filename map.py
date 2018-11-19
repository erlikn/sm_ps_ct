import tensorflow as tf
import numpy as np
import cv2
import sys
from os import listdir
from os.path import isfile, join
import os
import json
from scipy.stats import truncnorm

png_folder = "/Data2TB/SMATS/augmented/png/train/"

def generate():
    wrong_images = 0
    with open("/Data2TB/SMATS/augmented/png/combined_fixed.json") as f:
        jsondata = json.load(f)
    count = 0
    for i in range(0,len(jsondata['frames'])):
         filename = jsondata['frames'][i]['file']
         
         if os.path.exists(png_folder + filename):
            print(count)
            count += 1
            gt = []
            for j in range(0,len(jsondata['frames'][i]['annotations'])):
                if jsondata['frames'][i]['annotations'][j]['label'] == 'Head':
                    x = jsondata['frames'][i]['annotations'][j]['x']
                    y = jsondata['frames'][i]['annotations'][j]['y']
                    width = jsondata['frames'][i]['annotations'][j]['width']
                    height = jsondata['frames'][i]['annotations'][j]['height']
                    if(x + width > 640):
                        width = 640 - x
                    if(y + height > 480):
                        height = 480 - y
                    gt.append([x,y,width,height])
            if filename == "2018-03-15_11.18.13.788-resized-1.2-rotated--20.png" or filename == "2018-03-15_11.18.16.329-resized-1.2-rotated--20.png":
                image,save = add_gaussian(filename,gt)
                if(save):
                    cv2.imwrite("/Data2TB/SMATS/src/maps/" + filename, image)
                    x = 1
                else:
                    wrong_images += 1
            z = 1
    print("Wrong images: ", wrong_images)

def test():
    images = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]
    for image in images:
        if not os.path.exists("/Data2TB/SMATS/src/maps/" + image):
            x = 1


def add_gaussian(filename,gt):
    if filename == "2018-03-15_10.54.22.423-resized-1-rotated-0.png":
        x = 1
    image = cv2.imread(png_folder + filename, -1)
    gaussed_image = np.zeros((480,640,1), np.uint8)

    save = True

    for head in gt:
        x = head[0]
        y = head[1]
        width = head[2]
        height = head[3]
        center_x = x + (width/2)
        center_y = y + (height/2)
        if(width != 0 and height != 0):
            gx = 0.05* cv2.getGaussianKernel(width,int(width/4))
            gy = 0.05* cv2.getGaussianKernel(height,int(height/4))
            gy = np.transpose(gy)
            product = np.multiply(gx,gy)

    #        gx = cv2.getGaussianKernel(width,int(width/2))
    #        gy = cv2.getGaussianKernel(height,int(height/2))
    #        gy = np.transpose(gy)
    #        product = product*np.multiply(gx,gy)

            product = 255 * (product-np.min(product)) / (np.max(product)-np.min(product))
            #product = product.reshape([product.shape[0], product.shape[1], 1])
            #product2 = cv2.GaussianBlur(product,(int(height/2), int(width/2)+1), 30)
            #product2 = cv2.medianBlur(product.astype(np.float32),5)#(int((height+width)/2)))
            w = np.max(product)
            #kernel = np.ones([int(height/10),int(width/10)])/(int(height/10)+int(width/10))
            #product2 = cv2.filter2D(product,-1,kernel)
            #cv2.imshow('p',product)
            #cv2.waitKey()
            #cv2.imshow('p2',product2)
            #cv2.waitKey(0)
            product = product.flatten()
            count = 0
            xrange = x + width
            if xrange > 640:
                xrange = 640
            yrange = y + height
            if yrange > 480:
                yrange = 480
            for i in range(x,xrange):
                for j in range(y, yrange):
                    if(gaussed_image[j,i] == 0):
                        gaussed_image[j,i] = product[count]
                    else:
                        gaussed_image[j,i] = max(gaussed_image[j,i], product[count])
                    count += 1
            #gaussed_image[x:x+width + 1,y:y+height,0] = product

            #mask = A<B
            #A[mask]=B[mask]
            #cv2.imwrite("/Data2TB/SMATS/src/x.jpg", gaussed_image)
            #cv2.waitKey()
            t = 1
            #radius = width/2
            #if(height > width):
            #    radius = height/2
        else:
            print("WRONG IMAGE")
            save = False
    return gaussed_image, save
        

def main(_): 
    generate()

if __name__ == '__main__':
  tf.app.run()
