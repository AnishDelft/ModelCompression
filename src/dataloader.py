import os
import cv2
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import pickle
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data
from torch.autograd import Variable

from torchvision import transforms

## --------------------------------------- PASCAL VOC - v2 --------------------------------------- ##

class VOCDatasetv2(data.Dataset):

    def __init__(self, IMAGELIST_TXT, shape=None, shuffle=True, transform=None, target_transform=None, train=False, num_workers=1, seen=0, verbose=0):
        with open(IMAGELIST_TXT, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples         = len(self.lines)
        self.transform        = transform
        self.target_transform = target_transform
        self.train            = train
        self.shape            = shape

        self.seen             = seen
        self.num_workers      = num_workers
        self.verbose          = verbose

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), ' - [ERROR][dataloader.py] Index range error'
        imgpath = self.lines[index].rstrip()

        if self.verbose:
            print ('  -- [DEBUG][VOCDatasetv2] index : ', index)
            print ('  -- [DEBUG][VOCDatasetv2] imgpath : ', imgpath)
        
        if (0): # different images sizes
            if self.train and index % 64== 0:
                if self.seen < 4000*64:
                    width = 13*32
                    self.shape = (width, width)
                elif self.seen < 8000*64:
                    width = (random.randint(0,3) + 13)*32
                    self.shape = (width, width)
                elif self.seen < 12000*64:
                    width = (random.randint(0,5) + 12)*32
                    self.shape = (width, width)
                elif self.seen < 16000*64:
                    width = (random.randint(0,7) + 11)*32
                    self.shape = (width, width)
                else: # self.seen < 20000*64:
                    width = (random.randint(0,9) + 10)*32
                    self.shape = (width, width)

        if self.train:
            jitter     = 0.2
            hue        = 0.1
            saturation = 1.5 
            exposure   = 1.5

            img, label = getData(imgpath, self.shape, jitter, hue, saturation, exposure)
            label      = torch.from_numpy(label)

        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label   = torch.zeros(50*5)

            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                traceback.print_exc()
                tmp = torch.zeros(1,5)

            tmp = tmp.view(-1)
            tsz = tmp.numel()
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        
        return (img, label.float())

# --------------- DATA AUGMENTATIONS --------------- #
def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

# --------------- DATA AND LABELS --------------- #

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])

    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(int(truths.size/5), 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)

# Unused
def getLabels(labpath):
    max_boxes = 50
    label     = np.zeros((max_boxes,5))
    cc        = 0

    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)  # the values here are [norm_centre_x, norm_centre_y, norm_width, norm_height] wrt image-origin (top-left)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))

        for i in range(bs.shape[0]):
            label[cc] = bs[i]
            cc        += 1
            if cc >= 50:
                break
        
        label = np.reshape(label, (-1))
    
    return label

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)   # the values here are [norm_centre_x, norm_centre_y, norm_width, norm_height] wrt image-origin (top-left)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0

        for i in range(bs.shape[0]):
            if (1):
                # xmin, ymin
                x1 = bs[i][1] - bs[i][3]/2
                y1 = bs[i][2] - bs[i][4]/2
                # xmax, ymax
                x2 = bs[i][1] + bs[i][3]/2
                y2 = bs[i][2] + bs[i][4]/2
            
            if (1):
                x1 = min(0.999, max(0, x1 * sx - dx)) 
                y1 = min(0.999, max(0, y1 * sy - dy)) 
                x2 = min(0.999, max(0, x2 * sx - dx))
                y2 = min(0.999, max(0, y2 * sy - dy))
            
            if (1):
                bs[i][1] = (x1 + x2)/2 # x_centre
                bs[i][2] = (y1 + y2)/2 # y_centre
                bs[i][3] = (x2 - x1)   # width
                bs[i][4] = (y2 - y1)   # height

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue

            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def getData(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img                  = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label                = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)

    return img,label


## --------------------------------------- PASCAL VOC - v2 (voc_label.py) --------------------------------------- ##

# Step3 : converts box corners to centre and width,height
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    # centre of BBox
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    # Dimensions of BBox
    w = box[1] - box[0]
    h = box[3] - box[2]

    # Normalization
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x,y,w,h)

# Step2 : reads .xml () and converts to .txt
def convert_annotation(DATA_DIR, year, image_id):
    classes  = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    in_file  = open(os.path.join(DATA_DIR, 'VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id)))
    out_file = open(os.path.join(DATA_DIR,'VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id)), 'w')
    tree     = ET.parse(in_file)
    root     = tree.getroot()
    size     = root.find('size')
    w        = int(size.find('width').text)
    h        = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls       = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b      = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb     = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# Step1 : entry-point
def setup_VOC(DATA_DIR):

    sets    = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

    for year, image_set in sets:
        print (' - year : ', year, ' || image_set : ', image_set)
        if not os.path.exists(os.path.join(DATA_DIR, 'VOCdevkit/VOC%s/labels/'%(year))):
            os.makedirs(os.path.join(DATA_DIR, 'VOCdevkit/VOC%s/labels/'%(year)))

        image_ids = open(os.path.join(DATA_DIR, 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set))).read().strip().split()
        list_file = open(os.path.join(DATA_DIR, 'VOCdevkit/%s_%s.txt'%(year, image_set)), 'w')
        for image_id in image_ids:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(DATA_DIR, year, image_id))
            convert_annotation(DATA_DIR, year, image_id)
        list_file.close()

## --------------------------------------- PASCAL VOC - v1 --------------------------------------- ##

class YoloDataset(data.Dataset):
    
    def __init__(self, dir_data, file_annotations
                 , train
                 , image_size, grid_num
                 , flag_augm
                 , transform):
        
        self.dir_data   = dir_data
        self.dir_img    = os.path.join(dir_data, 'JPEGImages')
        self.train      = train
        self.transform  = transform
        
        self.fnames     = []
        self.boxes      = []
        self.labels     = []
        self.mean       = (123,117,104) # RGB ([How?])
        
        self.grid_num   = grid_num
        self.image_size = image_size
        self.flag_augm  = flag_augm

        self.verbose_aug = False

        with open(file_annotations) as f:
            for line in f.readlines():
                splited   = line.strip().split()
                self.fnames.append(splited[0])
                num_boxes = (len(splited) - 1) // 5
                box       = []
                label     = []
                for i in range(num_boxes):
                    x  = float(splited[1+5*i])
                    y  = float(splited[2+5*i])
                    x2 = float(splited[3+5*i])
                    y2 = float(splited[4+5*i])
                    c  = splited[5+5*i]
                    box.append([x,y,x2,y2])
                    label.append(int(c)+1)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))
                
        self.num_samples = len(self.boxes)
    
    def __getitem__(self,idx, verbose=0):
        fname  = self.fnames[idx]
        img    = cv2.imread(os.path.join(self.dir_img, fname), cv2.IMREAD_UNCHANGED)
        boxes  = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        
        if (0):
            print (' - fname :', fname)
            print (' - path : ', os.path.join(self.dir_img, fname))
            # plt.imshow(img)
            print (' - labels : ', labels)
        
        if self.train:
            if (self.flag_augm == 1):
                img = self.random_bright(img)
                img, boxes       = self.random_flip(img, boxes)
                img,boxes        = self.randomScale(img,boxes)
                img              = self.randomBlur(img)
                img              = self.RandomBrightness(img)
                img              = self.RandomHue(img)
                img              = self.RandomSaturation(img)
                img,boxes,labels = self.randomShift(img,boxes,labels)
                img,boxes,labels = self.randomCrop(img,boxes,labels)

        h,w,_  = img.shape
        
        boxes  /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img    = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        #img    = self.subMean(img,self.mean) 
        img    = cv2.resize(img,(self.image_size,self.image_size))
        target = self.encoder(boxes,labels) # 7x7x30
        for t in self.transform:
            img = t(img)
        return img,target
    
    def __len__(self):
        return self.num_samples
    
    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        
        target    = torch.zeros((self.grid_num, self.grid_num,30))
        cell_size = 1./self.grid_num
        wh        = boxes[:,2:] - boxes[:,:2]
        cxcy      = (boxes[:,2:] + boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample                       = cxcy[i]
            ij                                = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]),4]   = 1
            target[int(ij[1]),int(ij[0]),9]   = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            xy                                = ij*cell_size # The relative coordinates of the upper left corner of the matched mesh
            delta_xy                          = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2]  = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
            
        return target
    
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr  = bgr - mean
        return bgr
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            if self.verbose_aug:
                print (' - [AUG] : randomBrightness')
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            if self.verbose_aug:
                print (' - [AUG] : randomSaturation')
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            if self.verbose_aug:
                print (' - [AUG] : randomHue')
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            if self.verbose_aug:
                print (' - [AUG] : randomBlur')
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            if self.verbose_aug:
                print (' - [AUG] : randomShift')
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            if self.verbose_aug:
                print (' - [AUG] : randomScale')
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            if self.verbose_aug:
                print (' - [AUG] : randomCrop')
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels
    
    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            if self.verbose_aug:
                print (' - [AUG] : random_flip')
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes
    
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

    def display(self, X):
        plt.imshow(X.data.numpy().transpose(1,2,0))
    
    def display_anno(self, X,y):
        pass
    

# if __name__ == "__main__":
#     if (0):
#         pass
#         # dir_annotations  = 'yolo/data/VOCdevkit_trainval/VOC2007'
#         # file_annotations = 'yolo/data/VOCdevkit_trainval/VOC2007/anno_trainval.txt'
#         # image_size       = 448
#         # grid_num         = 14
#         # flag_augm        = 0
#         # train            = True
        
#         # YoloDatasetTrain = YoloDataset(dir_annotations, file_annotations
#         #                             , train
#         #                             , image_size, grid_num
#         #                             , flag_augm
#         #                             , transform = [transforms.ToTensor()] )
        
#         # print (' - Total Images: ', len(YoloDatasetTrain))
#         # if (1):
#         #     idx = np.random.randint(len(YoloDatasetTrain))
#         #     X,Y = YoloDatasetTrain[idx]
#         #     YoloDatasetTrain.display(X)
            
#         # DataLoaderTrain = DataLoader(YoloDatasetTrain, batch_size=1,shuffle=False,num_workers=0)
#         # train_iter = iter(DataLoaderTrain)
#         # for i in range(2):
#         #     img,target = next(train_iter)
#             # print(img,target) 
#     else:
#         TRAINLIST  = '../data/dataset/VOCdevkit/voc_train.txt'
#         TESTLIST   = '../data/dataset/VOCdevkit/2007_test.txt'
#         WIDTH      = 416
#         HEIGHT     = 416
#         BATCH_SIZE = 1 
#         if (0):
#             dataset_pascal = VOCDatasetv2(TRAINLIST, shape=(WIDTH, HEIGHT),
#                                 shuffle=False,
#                                 transform=transforms.Compose([
#                                     transforms.ToTensor(),
#                                 ]),
#                                 train=True,
#                                 seen=0, verbose=1)
#         else:
#             dataset_pascal = VOCDatasetv2(TESTLIST, shape=(WIDTH, HEIGHT),
#                                 shuffle=False,
#                                 transform=transforms.Compose([
#                                     transforms.ToTensor(),
#                                 ]),
#                                 train=False,
#                                 seen=0, verbose=1)
        
#         rand_idx = np.random.randint(len(dataset_pascal))
#         print (' - rand_idx : ', rand_idx)
#         X,Y = dataset_pascal[rand_idx]
#         print (' - X : ', X.shape, ' || dtype : ', X.dtype)
#         print (' - Y : ', Y.shape, ' || dtype : ', Y.dtype)
#         print (' - Y : ', Y)

#         dataset_pascal_loader = torch.utils.data.DataLoader(dataset_pascal, batch_size=BATCH_SIZE, shuffle=False)

#         for batch_idx, (data, target) in enumerate(dataset_pascal_loader):
#             if (batch_idx > 2):
#                 break
#             data, target = Variable(data), Variable(target)
#             print (' - [DEBUG] target : ', target)     
#             print (' - [DEBUG] target[target != 0.0]) : ', target[target != 0.0])