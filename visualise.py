import os
import sys
import torch
import cv2

from PIL import Image, ImageDraw

dir_main = os.path.abspath('/Users/Anish Mukherjee/Desktop/TU Delft/Q3/Deep Learning/Project/CS4180-DL/')
sys.path.append(dir_main)
print (' - In Path : ', sys.path[-1])

from src.nets2_utils import *
from src.nets import Darknet
from src.nets import *

USE_GPU = torch.cuda.is_available()
print (' - USE_GPU : ', USE_GPU)

if (1):
    cfgfile    = os.path.join(dir_main, 'data\\cfg\\github_pjreddie\\yolov2-voc.cfg')
    m = Darknet(cfgfile)
    weightfile = os.path.join(dir_main, 'data\\weights\\yolov2-voc.weights')
    namefile = ['aeroplane', 'bicycle','bird','boat','bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train', 'tvmonitor' ]    
    
    use_cuda = 1
    if use_cuda:
        m.cuda()
    
    img = Image.open('C:/Users/Anish Mukherjee/Desktop/TU Delft/Q3/Deep Learning/Project/CS4180-DL/data/VOCdevkit_test/VOC2007/JPEGImages/000001.jpg').convert('RGB')
    sized = img.resize((m.width, m.height))
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile = cv2.imread('C:/Users/Anish Mukherjee/Desktop/TU Delft/Q3/Deep Learning/Project/CS4180-DL/data/VOCdevkit_test/VOC2007/JPEGImages/000001.jpg'), (finish-start)))
            
    img = cv2.imread('C:/Users/Anish Mukherjee/Desktop/TU Delft/Q3/Deep Learning/Project/CS4180-DL/data/VOCdevkit_test/VOC2007/JPEGImages/000001.jpg')
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile = cv2.imread('C:/Users/Anish Mukherjee/Desktop/TU Delft/Q3/Deep Learning/Project/CS4180-DL/data/VOCdevkit_test/VOC2007/JPEGImages/000001.jpg'), (finish-start)))

    
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=namefile)
        

















