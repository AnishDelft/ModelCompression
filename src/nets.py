#encoding:utf-8
import cv2
import os
import sys
import pdb
import math
import time
import traceback
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.nn.functional as F

from src.pruning.weightPruning.layers import MaskedConv2d
from src.pruning.weightPruning.methods import quick_filter_prune, weight_prune
from src.pruning.weightPruning.utils import prune_rate, are_masks_consistent

from torch.autograd import Variable

runtime = 'online' # ['local', 'online']
if runtime == 'online':
    print (' - Online Runtime')
    from src.nets2_utils import *
elif runtime == 'local':
    print (' - Local Runtime')
    from nets2_utils import *
# from pruning.weightPruning.layers import MaskedLinear

torch.cuda.empty_cache()
USE_GPU = torch.cuda.is_available()

## --------------------------------------- YOLOV2 --------------------------------------- ##

## ----------------- YOLOV2:cfg
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

def print_cfg(blocks):
    print('layer     filters    size              input                output');
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters =[]
    out_widths =[]
    out_heights =[]
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)/2 if is_pad else 0
            width = (prev_width + 2*pad - kernel_size)/stride + 1
            height = (prev_height + 2*pad - kernel_size)/stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width/stride
            height = prev_height/stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width/stride
            height = prev_height/stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert(prev_width == out_widths[layers[1]])
                assert(prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'region':
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters,  filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))

## ----------------- YOLOV2 : load weights

def load_conv_old(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start

def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_conv_bn_old(buf, start, conv_model, bn_model, verbose=0):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    if (1):
        print ('      - conv weights : ', num_w)
        print ('      - bias weights : ', num_b)
        print ('      - bn_model.bias : ', bn_model.bias.shape)
        print ('      - bn_model.weight : ', bn_model.weight.shape)
    
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    
    if (1):
        print ('      - start : ', start)
        print ('      - size :  ', buf[start:start+num_w].shape)
        print ('      - shape :  ', conv_model.weight.data.shape)
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w 
    
    return start

def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]));   start = start + num_w 
    return start

def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)

def load_param(file, param):
    param.data.copy_(torch.from_numpy(
        np.fromfile(file, dtype=np.float32, count=param.numel()).reshape(param.shape)
    ))

def load_conv(file, conv_model):
    try:
        load_param(file, conv_model.bias)
        load_param(file, conv_model.weight)
    except:
        print ('  -- [Error] : load_conv()',)
        traceback.print_exc()
        pdb.set_trace()

def load_conv_bn(file, conv_model, bn_model):
    try:
        load_param(file, bn_model.bias)
        load_param(file, bn_model.weight)
        load_param(file, bn_model.running_mean)
        load_param(file, bn_model.running_var)
        load_param(file, conv_model.weight)
    except:
        print ('  -- [Error] : load_conv_bn()',)
        pdb.set_trace()

## ----------------- YOLOV2:loss
def build_targets(pred_boxes, target, anchors_list, anchors_cell, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    verbose    = 0
    MAX_BBOX   = 50
    THRESH_IOU = 0.5

    ## -------------------------- STEP 0 (init) -------------------------- ##
    if (1):
        nB          = target.size(0)
        nA          = anchors_cell
        nC          = num_classes
        anchor_step = int(len(anchors_list)/anchors_cell)
        
        coord_mask  = torch.zeros(nB, nA, nH, nW)
        # conf_mask   = torch.zeros(nB, nA, nH, nW) # torch.ones(nB, nA, nH, nW) * noobject_scale # [noobject_scale - why over here!!]
        conf_mask   = torch.ones(nB, nA, nH, nW) * noobject_scale
        cls_mask    = torch.zeros(nB, nA, nH, nW)

        tx          = torch.zeros(nB, nA, nH, nW) # True-x
        ty          = torch.zeros(nB, nA, nH, nW) # True-y
        tw          = torch.zeros(nB, nA, nH, nW) # True-w
        th          = torch.zeros(nB, nA, nH, nW) # True-h
        tconf       = torch.zeros(nB, nA, nH, nW) # True-conf (finally contains IoUs)
        tcls        = torch.zeros(nB, nA, nH, nW) # True-cls (finally contains class ids)

        nAnchors = nA*nH*nW
        nPixels  = nH*nW

        if verbose:
            print ('  -- [build_targets]')
            print ('  -- [build_targets] : Total Boxes : ', len(target[target != 0.0]) / 5)
            print ('  -- [build_targets] pred_boxes : ', pred_boxes.shape, ' (transposed from RegionLoss)') # [nAnchors,4]
            print ('  -- [build_targets] anchors_list : ', anchors_list)
            print ('  -- [build_targets] anchors_cell : ', anchors_cell)
            print ('  -- [build_targets] nAnchors : ', nAnchors)
            print ('  -- [build_targets] anchor_step : ', anchor_step) 
            print ('  -- [build_targets] nPixels : ', nPixels)
            print ('  -- [build_targets] sil_thresh : ', sil_thresh)

    ## -------------------------- STEP 1 (for confidence mask on the basis of IoU) -------------------------- ##
    if (1):
        for b in range(nB): # loop count = batch count
            cur_pred_boxes = pred_boxes[b*nAnchors : (b+1)*nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            for t in range(MAX_BBOX):     # max of 50 BBs per image in GTruth
                if target[b][t*5+1] == 0: # = target = [nB, 250] (a.k.a till all GTruth boxes are over)
                    break
                gx = target[b][t*5+1]*nW # (offset within grid cell in x-coordinates)
                gy = target[b][t*5+2]*nH # (offset within grid cell in y-coordinates)
                gw = target[b][t*5+3]*nW # ??
                gh = target[b][t*5+4]*nH # ??
                
                if verbose:
                    print ('  -- [build_targets][Box] : gx,gy,gw,gw : ', target[b][t*5+1], target[b][t*5+2], target[b][t*5+3], target[b][t*5+4])
                    print ('  -- [build_targets][Box] : gx,gy,gw,gw : ', gx,gy,gw,gw)
                
                cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
                tmp          = bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False)
                cur_ious     = torch.max(cur_ious, tmp)
            
            if (verbose):
                print ('  -- [build_targets] cur_pred_boxes : ', cur_pred_boxes.shape)
                print ('  -- [build_targets] cur_gt_boxes : ', cur_gt_boxes.shape)     # - [4, nAnchors]
                print ('  -- [build_targets] cur_ious : ', cur_ious.shape)
                # print (' -- [build_targets] : cur_ious.view(nA,nH,nW) > sil_thresh : ', cur_ious.view(nA,nH,nW) > sil_thresh)
            
            conf_mask[b][cur_ious.view(nA,nH,nW) > sil_thresh] = 0

    ## -------------------------- STEP 2 (for tx, ty, tw, th, coord_mask) -------------------------- ##
    if (1):
        pass
        # if seen < 12800:  # logic???
        #     if anchor_step == 4: # anchor_step = len(anchors) / 5
        #         tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
        #         ty = torch.FloatTensor(anchors).view(anchors_cell, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
        #     else:    
        #         tx.fill_(0.5)
        #         ty.fill_(0.5)
        #     tw.zero_()
        #     th.zero_()
        #     coord_mask.fill_(1)

    ## -------------------------- STEP 3 (builds tx, ty, tw, th, tconf, tcls) -------------------------- ##
    if (1):
        if verbose : print ('  -- [build_targets]')
        nGT      = 0
        nCorrect = 0
        for b in range(nB):
            for t in range(MAX_BBOX):
                if target[b][t*5+1] == 0:
                    break
                if verbose : print ('  -- [build_targets]')

                nGT      = nGT + 1
                best_iou = 0.0
                best_n   = -1
                min_dist = 10000

                ## -------------------------- STEP 3.1 (extract one BBox) -------------------------- ##
                if (1):
                    gx = target[b][t*5+1] * nW
                    gy = target[b][t*5+2] * nH
                    gw = target[b][t*5+3] * nW
                    gh = target[b][t*5+4] * nH
                    gi = int(gx)
                    gj = int(gy)
                    if verbose:
                        print ('  -- [build_targets][Box] : gx,gy,gw,gw : ', target[b][t*5+1], target[b][t*5+2], target[b][t*5+3], target[b][t*5+4])
                        print ('  -- [build_targets][Box] : gx,gy,gw,gw : ', gx,gy,gw,gw)
                        print ('  -- [build_targets][Box] : gi, gj : ', gi, gj)

                ## -------------------------- STEP 3.2 (find best anchor for that BBox) -------------------------- ##
                if (1):
                    gt_box = [0, 0, gw, gh] # notice how the gx, gy have not been filled in since we are interested only in the [best] anchor
                    for n in range(nA): #num_anchors
                        aw         = anchors_list[anchor_step*n]
                        ah         = anchors_list[anchor_step*n+1]
                        anchor_box = [0, 0, aw, ah]
                        iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                        if anchor_step == 4:
                            ax = anchors_list[anchor_step*n+2]
                            ay = anchors_list[anchor_step*n+3]
                            dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                        if iou > best_iou:
                            best_iou = iou
                            best_n = n
                        elif anchor_step==4 and iou == best_iou and dist < min_dist:
                            best_iou = iou
                            best_n = n
                            min_dist = dist
                    
                    if (verbose):
                        print ('  -- [build_targets][best_anchor] : best_n', best_n) # best_n out of anchors_cell
                        print ('  -- [build_targets][best_anchor] : best_iou', best_iou)

                if (1):
                
                    coord_mask[b][best_n][gj][gi] = 1   # = coord_mask = [nB, nA, nH, nW]
                    conf_mask[b][best_n][gj][gi]  = object_scale
                    cls_mask[b][best_n][gj][gi]   = 1
                    

                    # YOLOv2 - Section II - Direct Location Prediction
                    tx[b][best_n][gj][gi]         = target[b][t*5+1] * nW - gi # gx - gi #()
                    ty[b][best_n][gj][gi]         = target[b][t*5+2] * nH - gj # gy - gj
                    
                    # tw[b][best_n][gj][gi]         = math.log(gw/anchors_list[anchor_step*best_n])
                    # th[b][best_n][gj][gi]         = math.log(gh/anchors_list[anchor_step*best_n+1])
                    tw[b][best_n][gj][gi]         = gw/anchors_list[anchor_step*best_n]
                    th[b][best_n][gj][gi]         = gh/anchors_list[anchor_step*best_n+1]
                    
                    gt_box                        = [gx, gy, gw, gh]
                    pred_box                      = pred_boxes[b*nAnchors + best_n*nPixels + gj*nW + gi]
                    iou                           = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
                    tconf[b][best_n][gj][gi]      = iou # [Confused - why not 1!]
                    tcls[b][best_n][gj][gi]       = target[b][t*5]
                    if iou > THRESH_IOU:
                        nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):

    def __init__(self, num_classes=20, anchor_list=[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071], anchors_cell=5):
        super(RegionLoss, self).__init__()
        
        self.num_classes    = num_classes
        self.anchors        = anchor_list
        self.num_anchors    = anchors_cell
        self.anchor_step    = int(len(anchor_list)/anchors_cell)

        # - scaling values for loss function
        if (1):
            self.coord_scale    = 1
            self.noobject_scale = 1
            self.object_scale   = 1 # [5,1]
            self.class_scale    = 1
            self.thresh         = 0.6
        
        self.seen           = 0

        """
        References for Loss
            - [this]     https://github.com/marvis/pytorch-yolo2/blob/master/region_loss.py
            - [similiar] https://github.com/experiencor/keras-yolo2/blob/master/frontend.py
        """
        # print ('  -- [DEBUG][RegionLoss] self.num_anchors : ', self.num_anchors)

    def forward(self, output, target, verbose=0):
        verbose_shapes = 0
        verbose_loss   = 0
        CONF_THRESH    = 0.25

        ## -------------------------- STEP 0 (init) -------------------------- ## 
        if (1):
            nB = output.data.size(0)
            nA = self.num_anchors
            nC = self.num_classes
            nH = output.data.size(2)
            nW = output.data.size(3)
        
            if (verbose_shapes):
                #output : B x A x (4+1+num_classes) x H x W 
                print ('')
                print (' - [RegionLoss] -- coord_scale    : ', self.coord_scale)
                print (' - [RegionLoss] -- noobject_scale : ', self.noobject_scale)
                print (' - [RegionLoss] -- object_scale   : ', self.object_scale)
                print (' - [RegionLoss] -- class_scale    : ', self.class_scale)


                print (' - [RegionLoss] -- nB : ', nB)
                print (' - [RegionLoss] -- nA : ', nA)
                print (' - [RegionLoss] -- nC : ', nC)
                print (' - [RegionLoss] -- nH : ', nH)
                print (' - [RegionLoss] -- nW : ', nW)
                print (' - [RegionLoss] : output : ', output.shape) # = torch.Size([1, 125, 13, 13])
                print (' - [RegionLoss] : target : ',target.shape)  # = torch.Size([1, 250])
        
        ## -------------------------- STEP 1 (get all x,y,w,h,conf)-------------------------- ##
        if (1):
            output   = output.view(nB, nA, (5+nC), nH, nW)
            if (verbose):
                print (' - [RegionLoss] : output : ', output.shape) # = torch.Size([1, 5, 25, 13, 13])

            x    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW)) #[ since x,y are offsets within the grid cell we use a sigmoid]
            y    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW)) #[ since x,y are offsets within the grid cell we use a sigmoid]
            if (0):
                w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
                h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
            else:
                w    = torch.exp(output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW))
                h    = torch.exp(output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW))

            conf = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
            
            cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
            cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)

            if (verbose_shapes): # we get x,y,w,h,conf for each of the 5 anchors in each grid cell 
                print ('  -- [RegionLoss] : x : ', x.shape, ' || type : ', x.dtype) # = torch.Size([1, 5, 13, 13])  || type :  torch.float32
                print ('  -- [RegionLoss] : y : ', y.shape, ' || type : ', y.dtype) # = torch.Size([1, 5, 13, 13])  || type :  torch.float32
                print ('  -- [RegionLoss] : w : ', w.shape, ' || type : ', w.dtype) # = torch.Size([1, 5, 13, 13])  || type :  torch.float32
                print ('  -- [RegionLoss] : h : ', h.shape, ' || type : ', h.dtype) # = torch.Size([1, 5, 13, 13])  || type :  torch.float32
                print ('  -- [RegionLoss] : conf : ', conf.shape, ' || type : ', conf.dtype) # = torch.Size([1, 5, 13, 13])  || type :  torch.float32
                print ('  -- [RegionLoss] : cls : ', cls.shape, ' || type : ', cls.dtype)    # =  torch.Size([845, 20])  || type :  torch.float32

        ## -------------------------- STEP 2 (pred_boxes = grids and anchors ??) -------------------------- ##
        if (1):
            pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
            grid_x     = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
            grid_y     = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
            anchor_w   = torch.Tensor(self.anchors).view(nA, int(self.anchor_step)).index_select(1, torch.LongTensor([0])).cuda()
            anchor_h   = torch.Tensor(self.anchors).view(nA, int(self.anchor_step)).index_select(1, torch.LongTensor([1])).cuda()
            anchor_w   = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
            anchor_h   = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

            if (verbose_shapes):
                print ('  -- [RegionLoss] : pred_boxes : ', pred_boxes.shape)  # = torch.Size([4, 845])
                print ('  -- [RegionLoss] : grid_x : ', grid_x.shape)          # = torch.Size([845])
                print ('  -- [RegionLoss] : grid_y : ', grid_y.shape)          # = torch.Size([845])
                print ('  -- [RegionLoss] : anchor_w : ', anchor_w.shape)      # = torch.Size([845])
                print ('  -- [RegionLoss] : anchor_h : ', anchor_h.shape)      # = torch.Size([845])

            pred_boxes[0] = x.view(-1).data + grid_x
            pred_boxes[1] = y.view(-1).data + grid_y
            pred_boxes[2] = torch.exp(w.view(-1).data) * anchor_w
            pred_boxes[3] = torch.exp(h.view(-1).data) * anchor_h
            pred_boxes    = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4)) # = torch.Size([845,4])

        ## -------------------------- STEP 3 (get true values) -------------------------- ##
        if (1):
            nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                                nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
            cls_mask   = (cls_mask == 1)
            nProposals = int((conf > CONF_THRESH).sum().data.item())
            tx         = Variable(tx.cuda()) # True-x
            ty         = Variable(ty.cuda()) # True-y
            tw         = Variable(tw.cuda()) # True-w
            th         = Variable(th.cuda()) # True-h
            tconf      = Variable(tconf.cuda()) # True-conf
            tcls       = Variable(tcls.view(-1)[cls_mask.view(-1)].long().cuda()) # True-class
            if (verbose_shapes):
                print ('  -- [RegionLoss]')
                print ('  -- [RegionLoss] : tx : ', tx.shape, ' || type : ', tx.dtype) # = torch.Size([1, 5, 13, 13])  || type :  torch.float32
                print ('  -- [RegionLoss] : ty : ', ty.shape, ' || type : ', ty.dtype)
                print ('  -- [RegionLoss] : tw : ', tw.shape, ' || type : ', tw.dtype)
                print ('  -- [RegionLoss] : th : ', th.shape, ' || type : ', th.dtype)
                print ('  -- [RegionLoss] : tconf : ', tconf.shape, ' || type : ', tconf.dtype)
                print ('  -- [RegionLoss] : tcls : ', tcls.shape, ' || type : ', tcls.dtype) # = tls :  torch.Size([no_of_boxes])  || type :  torch.int64
                # print ('  -- [RegionLoss] tx : ', tx[tx != 0.0])
                # print ('  -- [RegionLoss] ty : ', ty[ty != 0.0])
                print ('  -- [RegionLoss] : tw    : ', tw[tw != 0.0])
                print ('  -- [RegionLoss] : th    : ', th[th != 0.0])
                print ('  -- [RegionLoss] : tconf : ', tconf[tconf != 0.0])
                print ('  -- [RegionLoss] : tcls  : ', tcls[tcls != 0.0])
                print ('  -- [RegionLoss] : coord_mask : ', coord_mask.shape, ' || type : ', coord_mask.dtype) 
                print ('  -- [RegionLoss] : conf_mask  : ', conf_mask.shape,  ' || type : ', conf_mask.dtype) 
                print ('  -- [RegionLoss] : cls_mask   : ', cls_mask.shape,   ' || type : ', cls_mask.dtype)
            
            
            coord_mask = Variable(coord_mask.cuda())
            conf_mask  = Variable(conf_mask.cuda().sqrt())
            cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())  
            if (verbose_shapes):
                print ('  -- [RegionLoss]')
                print ('  -- [RegionLoss] : coord_mask : ', coord_mask.shape, ' || type : ', coord_mask.dtype) # torch.Size([1, 5, 13, 13])  || type :  torch.float32
                print ('  -- [RegionLoss] : conf_mask  : ', conf_mask.shape,  ' || type : ', conf_mask.dtype)  # torch.Size([1, 5, 13, 13])  || type :  torch.float32
                print ('  -- [RegionLoss] : cls_mask   : ', cls_mask.shape,   ' || type : ', cls_mask.dtype)   # torch.Size([845, 20])  || type :  torch.uint8

        ## -------------------------- STEP 4 (losses) -------------------------- ##
        if (1):
            cls       = cls[cls_mask].view(-1, nC)
            if (1):
                loss_x    = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask  , tx*coord_mask)/2.0
                loss_y    = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask  , ty*coord_mask)/2.0
                loss_w    = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask  , tw*coord_mask)/2.0
                loss_h    = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask  , th*coord_mask)/2.0
                loss_conf = 1                * nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
                loss_cls  = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
                loss      = (loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls)/nB
            else:
                loss_x    = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask  , tx*coord_mask)/2.0
                loss_y    = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask  , ty*coord_mask)/2.0
                # Protip: adding a tiny epsilon where you’re dividing or taking square roots will probably do the trick.
                loss_w    = self.coord_scale * nn.MSELoss(size_average=False)(torch.sqrt(w*coord_mask), torch.sqrt(tw*coord_mask))/2.0
                loss_h    = self.coord_scale * nn.MSELoss(size_average=False)(torch.sqrt(h*coord_mask), torch.sqrt(th*coord_mask))/2.0
                loss_conf = 1                * nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
                loss_cls  = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls) # cls = [no_of_BBoxes, nC] || tcls = [no_of_BBoxes]
                loss      = (loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls)/nB

            if (verbose_loss):
                print ('  -- [RegionLoss] loss_x :', loss_x)
                print ('  -- [RegionLoss] loss_y :', loss_y)
                print ('  -- [RegionLoss] loss_w :', loss_w)
                print ('  -- [RegionLoss] loss_h :', loss_h)
                print ('  -- [RegionLoss] loss_conf :', loss_conf)
                print ('  -- [RegionLoss] loss_cls :', loss_cls)
                print ('  -- [RegionLoss] loss : ', loss, loss)
                print ('  -- [RegionLoss] Total GT Boxes : ', len(tcls))

                # print ('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data.item(), loss_y.data.item(), loss_w.data.item(), loss_h.data.item(), loss_conf.data.item(), loss_cls.data.item(), loss.data.item()))
                """
                - 3 loss elements
                    - Box Loss
                        - position loss  : [ [(x_true - x_predict)^2             + (y_true - y_predict)^2)         ]  * coord_mask ] * coord_scale
                        - dimension loss : [ [(sqrt(w_true) - sqrt(w_predict)^2) + (sqrt(h_true - sqrt(h_predict)))]  * coord_mask ] * coord_scale
                    - Confidence Loss
                        - [ (c_box_true - c_box_pred)^2 ] * conf_mask         * object_scale
                        - [ (c_box_true - c_box_pred)^2 ] * [mask_cell_noobj] * noobject_scale
                    - Class Loss
                        - [(class_true - class_predict)^2] * cls_mask * class_scale
                """
        
        # pdb.set_trace()
        return loss
        # return loss/nB

## ----------------- YOLOV2:other modules

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view((B, C, int(H/hs), hs, int(W/ws), ws)).transpose(3,4).contiguous()
        x = x.view((B, C, int(H/hs*W/ws), hs*ws)).transpose(2,3).contiguous()
        x = x.view((B, C, hs*ws, int(H/hs), int(W/ws))).transpose(1,2).contiguous()
        x = x.view((B, hs*ws*C, int(H/hs), int(W/ws)))
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

## ----------------- YOLOV2: main module
# support route shortcut and reorg
class Darknet(nn.Module):

    def __init__(self, cfgfile, verbose=0):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss   = self.models[len(self.models)-1]

        self.width  = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors     = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

            # if verbose:
            if 1:
                print ('  -- [DEBUG][Darknet] self.anchors : ', self.anchors)
                print ('  -- [DEBUG][Darknet] self.num_anchors : ', self.num_anchors)
                print ('  -- [DEBUG][Darknet] self.anchor_step : ', self.anchor_step)
                print ('  -- [DEBUG][Darknet] self.num_classes : ', self.num_classes)
                print ('  -- [DEBUG][Darknet] self.loss : ', self.loss)

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        # self.loss = None
        outputs = dict()
        for block in self.blocks:   
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x

            elif block['type'] == 'route':
                # print (' - Block : Route')
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x

            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x

            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None

            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))

        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id         = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters         = int(block['filters'])
                kernel_size     = int(block['size'])
                stride          = int(block['stride'])
                is_pad          = int(block['pad'])
                pad             = int((kernel_size-1)/2) if is_pad else 0
                activation      = block['activation']
                model           = nn.Sequential()
                
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), MaskedConv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    print ('  -- [DEBUG] Non-BN Block : ', block)
                    model.add_module('conv{0}'.format(conv_id), MaskedConv2d(prev_filters, filters, kernel_size, stride, pad))

                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)

            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)/loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def load_weights(self, weightfile):
        with open(weightfile, mode='rb') as f:
            major = np.fromfile(f, dtype=np.int32, count=1)
            minor = np.fromfile(f, dtype=np.int32, count=1)
            np.fromfile(f, dtype=np.int32, count=1)  # revision
            if major * 10 + minor >= 2 and major < 1000 and minor < 1000:
                np.fromfile(f, dtype=np.int64, count=1)  # seen
            else:
                np.fromfile(f, dtype=np.int32, count=1)  # seen

            ind = -2
            for block in self.blocks:
                if ind >= len(self.models):
                    break
                ind = ind + 1
                if block['type'] == 'net':
                    continue

                elif block['type'] == 'convolutional':
                    model           = self.models[ind]
                    batch_normalize = int(block['batch_normalize'])
                    if batch_normalize:
                        start = load_conv_bn(f, model[0], model[1])
                    else:
                        # print ('')
                        # print ('  -- [DEBUG] No batch norm in block!! : ', block)
                        start = load_conv(f, model[0])
                
                elif block['type'] == 'connected':
                    model = self.models[ind]
                    if block['activation'] != 'linear':
                        start = load_fc(f, model[0])
                    else:
                        start = load_fc(f, model)
                elif block['type'] == 'maxpool':
                    pass
                elif block['type'] == 'reorg':
                    pass
                elif block['type'] == 'route':
                    pass
                elif block['type'] == 'shortcut':
                    pass
                elif block['type'] == 'region':
                    pass
                elif block['type'] == 'avgpool':
                    pass
                elif block['type'] == 'softmax':
                    pass
                elif block['type'] == 'cost':
                    pass
                else:
                    print('unknown type %s' % (block['type']))

    def load_weights_old(self, weightfile):
        fp          = open(weightfile, 'rb')
        header      = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen   = self.header[3]
        buf         = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block_i, block in enumerate(self.blocks):
            print ('')
            print (' ----------------- block : ', block_i, '(start:',start,')')
            print (' || --------------> block : ', block)
            if start >= buf.size:
                break
            ind = ind + 1
            print (' || ---------------> self.models[ind] : ', self.models[ind])
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                try:
                    model           = self.models[ind]
                    batch_normalize = int(block['batch_normalize'])
                    if batch_normalize:
                        start = load_conv_bn(buf, start, model[0], model[1])
                    else:
                        start = load_conv(buf, start, model[0])
                except:
                    print (' - [Err] Block :', block_i)
                    traceback.print_exc()
                    import sys; sys.exit(1)
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

    def set_masks(self, masks):
        count = 0
        for m in self.modules():
            try:
                if m[0].name == 'MaskedConv2d':
                    m[0].set_mask(masks[count])
                    count += 1
            except:
                pass


def debug_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print ('  -- [DEBUG] : ', name, '\t  - \t', round(param.grad.data.sum().item(),3), '   [',param.shape,']')

def getYOLOv2(cfgfile, weightfile):
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    if USE_GPU:
        model.cuda()
    return model

if __name__ == '__main__':
    # testYOLOv1()
    cfg_file = 'yolov2-voc.cfg'
    weights_file = 'yolov2-voc.weights'
    #weights_file = 'yolov2-voc-pruned.weights'
    #testYOLOv2(cfg_file, weights_file)
    model = getYOLOv2(cfg_file, weights_file)
    if (0):
        for pruning_perc in [10.,30.,50.,70.,90.,99.]:
            masks = weight_prune(model, pruning_perc)
            print("Num masks: ", len(masks))
            model.set_masks(masks)
            prune_rate(model)
            print(are_masks_consistent(model, masks))
            #model.save_weights('yolov2-voc-filter-prune-%s.weights' % pruning_perc)

    