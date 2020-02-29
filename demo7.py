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

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.load_weights(weightfile)

    class_names = ['aeroplane', 'bicycle','bird','boat','bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train', 'tvmonitor' ]    

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.005, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


if __name__ == '__main__':
    
        dir_main = os.path.abspath('/Users/Anish Mukherjee/Desktop/TU Delft/Q3/Deep Learning/Project/CS4180-DL/')
        sys.path.append(dir_main)
        print (' - In Path : ', sys.path[-1])
        
        cfgfile = os.path.join(dir_main, 'data\\cfg\\github_pjreddie\\yolov2-voc.cfg')
        weightfile = os.path.join(dir_main, 'data\\weights\\yolov2.weights')
        imgfile = os.path.join(dir_main, 'data\\image\\000001.jpg') 
        detect_skimage(cfgfile, weightfile, imgfile)
