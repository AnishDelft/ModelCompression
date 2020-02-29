#encoding:utf-8
import os
import sys
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
from torch.autograd import Variable

# print (os.getcwd())
from src.nets2_utils import *
# from .nets2_utils import *

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
    load_param(file, conv_model.bias)
    load_param(file, conv_model.weight)

def load_conv_bn(file, conv_model, bn_model):
    load_param(file, bn_model.bias)
    load_param(file, bn_model.weight)
    load_param(file, bn_model.running_mean)
    load_param(file, bn_model.running_var)
    load_param(file, conv_model.weight)

## ----------------- YOLOV2:modelling
def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    # print ('')
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = int(len(anchors)/num_anchors)
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    # print (' -- [build_targets] conf_mask : ', conf_mask.shape)

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        # print (' -- [build_targets] cur_pred_boxes : ', cur_pred_boxes.shape)
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            cur_ious     = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        
        #tmpp = cur_ious>sil_thresh
        # print (' ------ b:', b, ' || cur_ious : ', cur_ious.shape, ' || sil_thresh : ', sil_thresh)
        # print (' ------ b:', b, ' || cur_ious.view(nA,nH,nW) : ', cur_ious.view(nA,nH,nW).shape)
        conf_mask[b][cur_ious.view(nA,nH,nW) > sil_thresh] = 0

    if seen < 12800:
       if anchor_step == 4:
           tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
           ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
       else:
           tx.fill_(0.5)
           ty.fill_(0.5)
       tw.zero_()
       th.zero_()
       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gx = gx.float()
            gy = target[b][t*5+2] * nH
            gy = gy.float()
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gw = gw.float()
            gh = target[b][t*5+4]*nH
            gh = gh.float()
            gt_box = [0, 0, gw, gh]
            for n in range(nA):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi]   = 1
            conf_mask[b][best_n][gj][gi]  = object_scale
            tx[b][best_n][gj][gi]         = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi]         = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi]         = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi]         = math.log(gh/anchors[anchor_step*best_n+1])
            # print (' -- t : ', t, ' || gt_box : ', gt_box, type(gt_box), ' || pred_box : ', pred_box.shape, type(pred_box))
            iou                           = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b][best_n][gj][gi]      = iou
            tcls[b][best_n][gj][gi]       = target[b][t*5]
            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = int(len(anchors)/num_anchors)
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

        print (' - [DEBUG][RegionLoss] self.num_anchors : ', self.num_anchors)

    def forward(self, output, target):
        # print ('')
        # print (' - [RegionLoss] : output : ', output.shape)
        # print (' - [RegionLoss] : target : ',target.shape)

        #output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        # print (' - [RegionLoss] -- nB : ', nB)
        # print (' - [RegionLoss] -- nA : ', nA)
        # print (' - [RegionLoss] -- nC : ', nC)
        # print (' - [RegionLoss] -- nH : ', nH)
        # print (' - [RegionLoss] -- nW : ', nW)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        # x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        x    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        # print (' - [RegionLoss] : pred_boxes : ',pred_boxes.shape)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, int(self.anchor_step)).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, int(self.anchor_step)).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

        # print (' - [RegionLoss] -- x.data : ', x.shape, ' || grid_x : ', grid_x.shape)
        # print (' - [RegionLoss] -- x.data : ', x.view(-1).shape, ' || grid_x : ', grid_x.shape)
        #pred_boxes[0] = x.data + grid_x
        pred_boxes[0] = x.view(-1).data + grid_x
        # print (' - [RegionLoss] -- y.data : ', y.shape, ' || grid_y : ', grid_y.shape)
        pred_boxes[1] = y.view(-1).data + grid_y
        # print (' - [RegionLoss] -- w.data : ', w.shape, ' || anchor_w : ', anchor_w.shape)
        pred_boxes[2] = torch.exp(w.view(-1).data) * anchor_w
        # print (' - [RegionLoss] -- h.data : ', h.shape, ' || anchor_h : ', anchor_h.shape)
        pred_boxes[3] = torch.exp(h.view(-1).data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        # print (' - [RegionLoss] : pred_boxes : ',pred_boxes.shape)
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask = (cls_mask == 1)

        # print ('')
        # print (' -- conf : ', conf.shape, ' || (conf > 0.25)', (conf > 0.25).sum().data.item())
        # nProposals = int((conf > 0.25).sum().data[0])
        nProposals = int((conf > 0.25).sum().data.item())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        # print (' -- tcls : ', tcls.shape)
        # print (' -- tcls : ', tcls.view(-1).shape)
        # print (' -- cls_mask : ', cls_mask.shape)
        # import pdb; pdb.set_trace()
        # # tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())
        tcls  = Variable(tcls.view(-1)[cls_mask.view(-1)].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        # print ('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data.item(), loss_y.data.item(), loss_w.data.item(), loss_h.data.item(), loss_conf.data.item(), loss_cls.data.item(), loss.data.item()))
        return loss

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

# support route shortcut and reorg
class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss = None
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
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad    = int((kernel_size-1)/2) if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
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
                    model = self.models[ind]
                    batch_normalize = int(block['batch_normalize'])
                    if batch_normalize:
                        start = load_conv_bn(f, model[0], model[1])
                    else:
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

def getYOLOv2(cfgfile, weightfile):
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    if USE_GPU:
        model.cuda()
    return model

def testYOLOv2():

    cfgfile    = 'data/cfg/github_pjreddie/yolov2-voc.cfg'
    weightfile = 'data/weights/github_pjreddie/yolov2-voc.weights'
    model      = getYOLOv2(cfgfile, weightfile) 
    print (' - 1. Model is loaded!')

    imgdir      = 'data/dataset/yolo_samples'
    namesfile   = 'data/dataset/voc.names'
    class_names = load_class_names(namesfile)
    for each in ['dog.jpg', 'eagle.jpg',  'giraffe.jpg',  'horses.jpg',  'person.jpg',  'scream.jpg']:
        try:
            print ('')
            imgfile = os.path.join(imgdir, each)        
            img     = Image.open(imgfile).convert('RGB')
            sized   = img.resize((model.width, model.height))

            start  = time.time()
            boxes  = do_detect(model, sized, 0.5, 0.4, USE_GPU)
            finish = time.time()
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))
            plot_boxes(img, boxes, os.path.join(imgdir, '_' + each), class_names)
        except:
            traceback.print_exc()
            pass

class TinyYoloNet(nn.Module):
    def __init__(self):
        super(TinyYoloNet, self).__init__()
        self.seen = 0
        self.num_classes = 20
        self.anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
        self.num_anchors = len(self.anchors)/2
        num_output = (5+self.num_classes)*self.num_anchors
        self.width = 160
        self.height = 160

        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)
        self.cnn = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d( 3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),

            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),

            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, 2)),

            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),
            ('pool6', MaxPoolStride1()),

            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8
            ('conv8', nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(1024)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('output', nn.Conv2d(1024, num_output, 1, 1, 0)),
        ]))

    def forward(self, x):
        x = self.cnn(x)
        return x

    def print_network(self):
        print(self)

    def load_weights(self, path):
        #buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
        buf = np.fromfile(path, dtype = np.float32)
        start = 4
        
        start = load_conv_bn(buf, start, self.cnn[0], self.cnn[1])
        start = load_conv_bn(buf, start, self.cnn[4], self.cnn[5])
        start = load_conv_bn(buf, start, self.cnn[8], self.cnn[9])
        start = load_conv_bn(buf, start, self.cnn[12], self.cnn[13])
        start = load_conv_bn(buf, start, self.cnn[16], self.cnn[17])
        start = load_conv_bn(buf, start, self.cnn[20], self.cnn[21])
        
        start = load_conv_bn(buf, start, self.cnn[24], self.cnn[25])
        start = load_conv_bn(buf, start, self.cnn[27], self.cnn[28])
        start = load_conv(buf, start, self.cnn[30])

## --------------------------------------- YOLOV1 --------------------------------------- ##


def getYOLOv1Self(name=''):

    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    if name != '':
        myYOLO            = YOLOv1Self(name, cfg['D'], batch_norm=True)
        VGG               = models.vgg16_bn(pretrained=True)
        state_dict_VGG    = VGG.state_dict()
        state_dict_myYOLO = myYOLO.state_dict()
        
        for k in state_dict_VGG.keys():
            if k in state_dict_myYOLO.keys() and k.startswith('features'):
                state_dict_myYOLO[k] = state_dict_VGG[k]
        myYOLO.load_state_dict(state_dict_myYOLO)
        return myYOLO

    else:
        print (' - Pass a name for your model')
        sys.exit(1)

class YOLOv1Self(nn.Module):

    def __init__(self, name, cfg, batch_norm, image_size=448):
        super(YOLOv1Self, self).__init__()
        self.name       = name
        self.features   = self.getFeatureLayers(cfg, batch_norm)
        self.linear1    = MaskedLinear(512 * 7 * 7, 4096)
        self.linear2    = MaskedLinear(4096, 1470)
        self.classifier = nn.Sequential( # add the regression part to the features
            # nn.Linear(512 * 7 * 7, 4096),
            self.linear1,
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 1470),
            self.linear2,
        )
        self._initialize_weights()
        self.image_size = image_size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        x = x.view(-1,7,7,30)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def getFeatureLayers(self, cfg, batch_norm=False):
        if (1):
            params_in_channels  = 3
            params_conv_stride  = 1
            params_conv_size    = 3 
            params_first_flag   = True
            params_pool_stride  = 2
            params_pool_kernel  = 2

        layers = []
        for item in cfg:
            params_conv_stride = 1
            if (item == 64 and params_first_flag):
                params_conv_stride = 2
                params_first_flag  = False

            if item == 'M': # max-pooling
                layers += [nn.MaxPool2d(kernel_size=params_pool_kernel, stride=params_pool_stride)]
            else:
                params_kernels = item
                conv2d = nn.Conv2d(params_in_channels, params_kernels, kernel_size=params_conv_size, stride=params_conv_stride, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(item), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                params_in_channels = item
        return nn.Sequential(*layers)

    def set_masks(self, masks):
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])

def testYOLOv1():
    if (0):
        net = getYOLOv1Self()
        img = torch.rand(1,3,448,448)
        img = Variable(img)
        output = net(img)
        print(output.size())
    else:
        pass

# if __name__ == '__main__':
#     # testYOLOv1()
#     # testYOLOv2()
#     # blocks = parse_cfg('/home/strider/Work/Netherlands/TUDelft/1_Courses/Sem2/DeepLearning/Project/repo1/data/cfg/github_pjreddie/yolov2-voc.cfg',1)
#     blocks = parse_cfg('/home/strider/Work/Netherlands/TUDelft/1_Courses/Sem2/DeepLearning/Project/repo1/data/cfg/github_pjreddie/yolov1.cfg',1)
#     model  = create_network(self.blocks) # merge conv, bn,leaky
    



"""
URLs
    - YOLOv1 
        - https://pjreddie.com/darknet/yolov1/
            - wget http://pjreddie.com/media/files/yolov1/yolov1.weights (~800MB) [trained on 2007 train/val+ 2012 train/val]
            - https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov1.cfg (output = 1715 = 7x7x(3x5 + 20) ) 
                - Locally Connected FC (https://discuss.pytorch.org/t/how-does-pytorch-implement-local-convolutional-layer-local-connected-layer/4316)
                    - https://github.com/pjreddie/darknet/issues/876
                    - https://github.com/pytorch/pytorch/issues/499
                    - https://github.com/pytorch/pytorch/compare/master...1zb:conv-local
                    - Theory : https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec11_handout.pdf
                - Locally Connected FC (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LocallyConnected2D)
    - YOLOv2    
        - https://pjreddie.com/darknet/yolov2/
            - https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg [416 x 416]
            - wget https://pjreddie.com/media/files/yolov2-voc.weights (~MB) [trained on 2007 train/val+ 2012 train/val]
        - On other githubs
            - wget http://pjreddie.com/media/files/yolo.weights
            - wget https://pjreddie.com/media/files/yolo-voc.weights
    - Dataset
        - <> 

Results
    - YOLOv1 (inside YOLO paper)
        - (2007 + 2012) = 63.4 mAP
        - (2007 + 2012) = 66.4 mAP (VGG-16)
    - repo1 (https://github.com/yxlijun/tensorflow-yolov1)
        - (2007 + 2012) = 65.3 mAP (VGG-16)
        - (2007 + 2012) = 66.12 mAP (VGG-19)
        - (2007 + 2012) = 65.23 mAP (Resnet)

Pre-trained Weights
    - YOLOv1
        - repo1 : https://github.com/dshahrokhian/YOLO_tensorflow
            - YOLO_small.ckpt
        - repo1 : https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html
            - convert from pjreddie --> tensorflow --> convert to pytorch

Converter
    - https://github.com/microsoft/MMdnn
    - https://github.com/marvis/pytorch-caffe-darknet-convert
    - https://github.com/AceCoooool/YOLO-pytorch/blob/master/tools/yad2t.py
    - https://github.com/thtrieu/darkflow.git
        - git clone https://github.com/thtrieu/darkflow.git
        - cd darkflow
        - pip install -e .
        - wget http://pjreddie.com/media/files/yolov1/yolov1.weights
        - flow --model cfg/v1.1/yolov1.cfg --load ../../repo1/data/weights/github_pjreddie/yolov1.weights --savepb
        - Colab
            - # ! git clone https://github.com/thtrieu/darkflow.git
                # ! cd darkflow && pip install -e .
                # ! wget http://pjreddie.com/media/files/yolov1/yolov1.weights
                # ! flow -h
                # ! ls -l
                # ! flow --model darkflow/cfg/v1.1/yolov1.cfg --load yolov1.weights --savepb

Random
    - https://github.com/happyjin/pytorch-YOLO/blob/master/network.py
    - https://github.com/kevin970401/pytorch-YOLO-v1/blob/master/models/yolo.py
"""

"""
We train the network for about 
 - 135 epochs on the train-ing and validation data sets from PASCAL VOC 2007 and 2012. 
 - When testing on 2012 we also include the VOC 2007 test data for training. 
 - Throughout training we use a 
    - batch size of 64, a momentum of 0.9 and a decay of 0.0005. 
 - Our learning rate schedule is as follows: 
    - For the first epochs we slowly raise the learning rate from 10−3 to 10−2. 
    - If we start at a high learning rate our model often diverges due to unstable gradients. 
    - We continue training with 
        - 10−2 for 75 epochs
        - then 10−3 for 30 epochs
        - and finally 10−4 for 30 epochs
"""


