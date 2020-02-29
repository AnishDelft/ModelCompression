import sys
import pdb
import os
import time
import math
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.autograd import Variable

import struct # get_image_size
import imghdr # get_image_size

import matplotlib.pyplot as plt


def parse_cfg(cfgfile, verbose=0):
    blocks = []
    fp     = open(cfgfile, 'r')
    block  =  None
    line   = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue

        elif line[0] == '[':
            if block:
                if verbose:
                    print ('')
                    print (' - block : ', block)
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def bbox_iou(box1, box2, x1y1x2y2=True):
    try:
        # print (' ------------- [bbox_iou] box1 : ', box1, ' || box2 : ', box2)
        if x1y1x2y2:
            mx = min(box1[0], box2[0])
            Mx = max(box1[2], box2[2])
            my = min(box1[1], box2[1])
            My = max(box1[3], box2[3])
            w1 = box1[2] - box1[0]
            h1 = box1[3] - box1[1]
            w2 = box2[2] - box2[0]
            h2 = box2[3] - box2[1]
        else:
            mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
            Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
            my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
            My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
            w1 = box1[2]
            h1 = box1[3]
            w2 = box2[2]
            h2 = box2[3]
        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh
        carea = 0
        if cw <= 0 or ch <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        return carea/uarea
    except:
        traceback.print_exc()

def bbox_ious(boxes1, boxes2, x1y1x2y2=True): # expects boxes1 = [4,?] || boxes2 = [4,?]
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]

    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea

    return carea/uarea

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

# ----------------------------------- predict.py -----------------------------------

def get_region_boxes(output, CONF_THRESH, num_classes, anchors_list, anchors_cell, only_objectness=1, validation=False):
    verbose = 0
    ## -------------------------- STEP 0 (init)-------------------------- ##    
    if (1):
        if output.dim() == 3:
            output = output.unsqueeze(0)
        assert(output.size(1) == (5+num_classes)*anchors_cell)
        anchor_step = int(len(anchors_list)/anchors_cell)
        batch       = output.size(0)    
        h           = output.size(2)
        w           = output.size(3)
        if (verbose):
            print ('  -- [DEBUG][get_region_boxes] output : ', output.shape)

    ## -------------------------- STEP 1.1 (get xs, ys, ws, hs, confs, cls)-------------------------- ##
    if (1):
        
        output      = output.view(batch*anchors_cell, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*anchors_cell*h*w)
        if (verbose):
            print ('  -- [DEBUG][get_region_boxes] output : ', output.shape)

        # (x,y) - apply the same sigmoid(x), sigmoid(y) to every cell in the grid
        grid_x      = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*anchors_cell, 1, 1).view(batch*anchors_cell*h*w).cuda()
        grid_y      = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*anchors_cell, 1, 1).view(batch*anchors_cell*h*w).cuda()
        xs          = torch.sigmoid(output[0]) + grid_x
        ys          = torch.sigmoid(output[1]) + grid_y

        # (w,h) - trying every combination of anchors
        anchor_w    = torch.Tensor(anchors_list).view(anchors_cell, anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h    = torch.Tensor(anchors_list).view(anchors_cell, anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w    = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*anchors_cell*h*w).cuda()
        anchor_h    = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*anchors_cell*h*w).cuda()
        ws          = torch.exp(output[2]) * anchor_w
        hs          = torch.exp(output[3]) * anchor_h

        box_confs   = torch.sigmoid(output[4])

        cls_confs                  = torch.nn.Softmax(dim=1)(Variable(output[5:5+num_classes].transpose(0,1))).data
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs              = cls_max_confs.view(-1)
        cls_max_ids                = cls_max_ids.view(-1)
    
    ## -------------------------- STEP 1.2 (bring to cpu)-------------------------- ##
    if (1):
        sz_hw         =  h*w
        sz_hwa        = sz_hw*anchors_cell
        box_confs     = convert2cpu(box_confs)
        cls_max_confs = convert2cpu(cls_max_confs)
        cls_max_ids   = convert2cpu_long(cls_max_ids)
        xs            = convert2cpu(xs)
        ys            = convert2cpu(ys)
        ws            = convert2cpu(ws)
        hs            = convert2cpu(hs)
        if validation:
            cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    
    ## -------------------------- STEP 2 (loop over all)-------------------------- ##
    all_boxes   = []
    if (1):
        for b in range(batch):
            boxes = []
            for cy in range(h):
                for cx in range(w):
                    for i in range(anchors_cell): # loop over [batches, grid_y, grid_x, anchors_i]
                        ind      = b * sz_hwa + i*sz_hw + cy*w + cx
                        box_conf =  box_confs[ind]

                        ## Step 2.1 - Box Confidence
                        if only_objectness:
                            conf =  box_confs[ind]
                        else:
                            conf = box_confs[ind] * cls_max_confs[ind] 
        
                        if conf > CONF_THRESH:
                            bcx          = xs[ind]  # b stands for bounding-box
                            bcy          = ys[ind]
                            bw           = ws[ind]
                            bh           = hs[ind]
                            cls_max_conf = cls_max_confs[ind]
                            cls_max_id   = cls_max_ids[ind]
                            box          = [bcx/w, bcy/h, bw/w, bh/h, box_conf, cls_max_conf, cls_max_id]

                            if (not only_objectness) and validation:
                                for c in range(num_classes):
                                    tmp_conf = cls_confs[ind][c]
                                    if c != cls_max_id and box_confs[ind]*tmp_conf > CONF_THRESH:
                                        box.append(tmp_conf)
                                        box.append(c)

                            boxes.append(box)

            all_boxes.append(boxes)
        
    return all_boxes

def nms(boxes, NMS_THRESH):
    if len(boxes) == 0:
        return boxes

    ## -------------------------- STEP 1 (??)-------------------------- ##
    if (1):
        box_confs = torch.zeros(len(boxes))
        for i in range(len(boxes)):
            box_confs[i] = 1-boxes[i][4]  ## why 1 - det_conf            

    ## -------------------------- STEP 1 (??)-------------------------- ##
    if (1):
        _,sortIds = torch.sort(box_confs)
        out_boxes = []
        for i in range(len(boxes)):
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
                for j in range(i+1, len(boxes)):
                    box_j = boxes[sortIds[j]]
                    if bbox_iou(box_i, box_j, x1y1x2y2=False) > NMS_THRESH: #comparing (box_i, box_j) and .... ?? 
                        #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                        box_j[4] = 0
    return out_boxes

# ----------------------------------- ?? -----------------------------------

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None, verbose=0):
    import cv2
    colors = [
            [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], 
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], 
            [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]]
    # def get_color(c, x, max_val):
    #     ratio = float(x)/max_val * 5
    #     i = int(math.floor(ratio))
    #     j = int(math.ceil(ratio))
    #     ratio = ratio - i
    #     r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
    #     return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    # print ('  -- [DEBUG][plot_boxes_cv2] boxes : ', len(boxes))
    for i in range(len(boxes)):
        box = boxes[i]
        
        x1 = int(round((box[0].item() - box[2].item()/2.0) * width))
        y1 = int(round((box[1].item() - box[3].item()/2.0) * height))
        x2 = int(round((box[0].item() + box[2].item()/2.0) * width))
        y2 = int(round((box[1].item() + box[3].item()/2.0) * height))
        
        # if color:
        #     rgb = color
        # else:
        #     rgb = (255, 0, 0)

        if len(box) >= 7 and class_names:
            box_conf   = box[4]
            cls_conf   = box[5]
            cls_id     = box[6]
            cls_color  = colors[cls_id]
            class_name = class_names[cls_id]
            if (0):
                cls_font   = cv2.FONT_HERSHEY_PLAIN
                cls_font_scale = 1.0
                cls_font_thick = 1
            else:
                cls_font   = cv2.FONT_HERSHEY_SIMPLEX
                cls_font_scale = 0.6
                cls_font_thick = 1

            if verbose:
                print('  -- [DEBUG][plot_boxes_cv2] %s: %.3f || %.3f' % (class_names[cls_id], box_conf, cls_conf))
            
            img = cv2.rectangle(img, (x1,y1), (x2,y2), cls_color, 3)    
            text_size, baseline = cv2.getTextSize(class_name, cls_font, cls_font_scale, cls_font_thick)
            if (y1-text_size[1] > 0):
                p1                  = (x1, y1- text_size[1])
                img = cv2.rectangle(img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), cls_color, -1)
                img = cv2.putText(img, class_name, (x1,y1), cls_font, cls_font_scale, (255,255,255), cls_font_thick,cv2.LINE_AA)
            else:
                p1                  = (x2 - text_size[0], y2)
                print (p1)
                img = cv2.rectangle(img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), cls_color, -1)
                img = cv2.putText(img, class_name, (x2 - text_size[0],y2), cls_font, cls_font_scale, (255,255,255), cls_font_thick,cv2.LINE_AA)
        
    if 0:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)

    # img  = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=[255, 255, 255])
    # img  = cv2.resize(img,(416,416))
    return img

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1, verbose=0):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    output = model(img)
    output = output.data
    #for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    #print('')
    t3 = time.time()

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)[0]
    
    #for j in range(len(boxes)):
    #    print(boxes[j])
    t4 = time.time()

    boxes = nms(boxes, nms_thresh)
    if verbose:
        print ('  -- [DEBUG][do_detect] : boxes (post-nms) :', len(boxes))
        print ('  -- [DEBUG][do_detect] : boxes :', len(boxes))
    t5 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('             nms : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t0))
        print('-----------------------------------')
    return boxes

def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print (' - cls_id : ', cls_id)
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

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

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets
      
def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count(b'\n')
    thefile.close( )
    return count

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24: 
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2 
                ftype = 0 
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2 
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))


if __name__ == "__main__":
    file = '/home/strider/Work/Netherlands/TUDelft/1_Courses/Sem2/DeepLearning/Project/repo1/data/dataset/voc.names'
    class_names = load_class_names(file)
    print (class_names)