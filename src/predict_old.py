import os
import cv2
import pdb
import tqdm
import traceback
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import requests

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if (0):
    from src.nets import *
from src.dataloader import * 
from src.nets2_utils import *

# from google.colab.patches import cv2_imshow

USE_GPU = torch.cuda.is_available()

class PASCALVOCEval():

    def __init__(self, MODEL, MODEL_CFGFILE, MODEL_WEIGHTFILE, MODEL_LOSS
    , PASCAL_DIR, EVAL_IMAGELIST, EVAL_OUTPUTDIR, EVAL_PREFIX, EVAL_OUTPUTDIR_PKL
    , LOGGER='', LOGGER_EPOCH=-1):
        self.MODEL            = MODEL
        self.MODEL_CFGFILE    = MODEL_CFGFILE
        self.MODEL_WEIGHTFILE = MODEL_WEIGHTFILE
        self.MODEL_LOSS       = MODEL_LOSS
        self.PASCAL_DIR       = PASCAL_DIR
        self.EVAL_IMAGELIST   = EVAL_IMAGELIST
        self.EVAL_OUTPUTDIR   = EVAL_OUTPUTDIR
        self.EVAL_PREFIX      = EVAL_PREFIX
        self.EVAL_OUTPUTDIR_PKL = EVAL_OUTPUTDIR_PKL
        self.LOGGER             = LOGGER
        self.LOGGER_EPOCH       = LOGGER_EPOCH
        self.USE_GPU            = torch.cuda.is_available()

        self.VOC_CLASSES = (    # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        
        self.VOC_CLASSES_ = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor') 
        self.VOC_YEAR = '2007'
        
    def predict(self, BATCH_SIZE=2, CONF_THRESH=0.005,NMS_THRESH=0.45):
        # CONF_THRESH=0.25,NMS_THRESH=0.45, IOU_THRESH    = 0.5

        if (1):
            if self.MODEL == '' or self.MODEL == None:
                print (' - 1. Loading model : ', self.MODEL_WEIGHTFILE)
                self.MODEL = getYOLOv2(self.MODEL_CFGFILE, self.MODEL_WEIGHTFILE)
            self.MODEL.eval()

        if (1):
            with open(self.EVAL_IMAGELIST) as fp:
                tmp_files   = fp.readlines()
                valid_files = [item.rstrip() for item in tmp_files]
            eval_dataset = VOCDatasetv2(self.EVAL_IMAGELIST, shape=(self.MODEL.width, self.MODEL.height),
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]))
            eval_batchsize = BATCH_SIZE
            kwargs = {'num_workers': 4, 'pin_memory': True}
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=eval_batchsize, shuffle=False, **kwargs) 

        if (1): 
            fps = [0]*self.MODEL.num_classes
            if not os.path.exists(self.EVAL_OUTPUTDIR):
                os.mkdir(self.EVAL_OUTPUTDIR)
            for i in range(self.MODEL.num_classes):
                buf = '%s/%s%s.txt' % (self.EVAL_OUTPUTDIR, self.EVAL_PREFIX, self.VOC_CLASSES[i])
                fps[i] = open(buf, 'w')
    
        lineId = -1
        verbose = 1
        
        with torch.no_grad():
            val_loss_total       = 0.0
            with tqdm.tqdm_notebook(total = len(eval_loader)*BATCH_SIZE) as pbar:
                
                for batch_idx, (data, target) in enumerate(eval_loader):
                    pbar.update(BATCH_SIZE)

                    t1 = time.time()
                    if self.USE_GPU:
                        data   = data.cuda()
                        target = target.cuda()                        
                    data, target = Variable(data), Variable(target)
                    output       = self.MODEL(data).data
                    t2 = time.time()

                    # if self.LOGGER != '':
                    #     if self.MODEL_LOSS != None:
                    #         try:
                    #             print (' - [DEBUG] region_loss : ', self.MODEL_LOSS)
                    #             val_loss     = self.MODEL_LOSS(output, target)
                    #             val_loss_total += val_loss.data
                    #         except:
                    #             traceback.print_exc()
                    #             pdb.set_trace()

                    batch_boxes = get_region_boxes(output, CONF_THRESH, self.MODEL.num_classes, self.MODEL.anchors, self.MODEL.num_anchors, 0, 1)
                    t3 = time.time()

                    for i in range(output.size(0)):
                        lineId        = lineId + 1
                        fileId        = os.path.basename(valid_files[lineId]).split('.')[0]
                        width, height = get_image_size(valid_files[lineId])
                        # print(valid_files[lineId])
                        boxes = batch_boxes[i]
                        boxes = nms(boxes, NMS_THRESH)
                        for box in boxes:
                            x1 = (box[0] - box[2]/2.0) * width
                            y1 = (box[1] - box[3]/2.0) * height
                            x2 = (box[0] + box[2]/2.0) * width
                            y2 = (box[1] + box[3]/2.0) * height

                            det_conf = box[4]
                            for j in range(int((len(box)-5)/2)):
                                cls_conf = box[5+2*j]
                                cls_id   = box[6+2*j]
                                prob     = det_conf * cls_conf
                                fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))
                    t4 = time.time()

                    if (verbose):
                        print ('  -- [DEBUG][PASCALVOCEval] Total time  : ', round(t4 - t1,2))
                        print ('  -- [DEBUG][PASCALVOCEval] output time :  ', round(t2 - t1,2))
                        print ('  -- [DEBUG][PASCALVOCEval] boxes time  :  ', round(t3 - t2,2))
                        print ('  -- [DEBUG][PASCALVOCEval] file write  :  ', round(t4 - t3,2))

        if self.LOGGER != '':
            if self.MODEL_LOSS != None:
                self.LOGGER.save_value('Total Loss', 'Val Loss', self.LOGGER_EPOCH+1, val_loss_total / len(eval_loader))

        for i in range(self.MODEL.num_classes):
            fps[i].close()
        
        self._do_python_eval()
    
    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def voc_eval(self, detpath,
                annopath,
                imagesetfile,
                classname,
                cachedir,
                ovthresh=0.5,
                use_07_metric=False):
        """rec, prec, ap = voc_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])

        Top level function that does the PASCAL VOC evaluation.

        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # print (' - cachefile : ', cachefile)
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(annopath.format(imagename))
                # if i % 100 == 0:
                #     # print ('Reading annotation for {0}/{1}'.format(i + 1, len(imagenames))
                #     print ('Reading annotation for ', i+1, '/', len(imagenames))
            # save
            # print ('Saving cached annotations to {0}'.format(cachefile))

            with open(cachefile, 'wb') as f:
                cPickle.dump(recs, f)

        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = cPickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap

    def _do_python_eval(self):
        print ('  -- Reading predictions from : ', self.EVAL_OUTPUTDIR, '/',self.EVAL_PREFIX,'*')
        res_prefix   = os.path.join(self.EVAL_OUTPUTDIR, self.EVAL_PREFIX)
        filename     = res_prefix + '{:s}.txt'
        annopath     = os.path.join(self.PASCAL_DIR, 'VOC' + self.VOC_YEAR, 'Annotations','{:s}.xml')
        imagesetfile = os.path.join(self.PASCAL_DIR, 'VOC' + self.VOC_YEAR,'ImageSets','Main','test.txt')
        cachedir     = os.path.join(self.PASCAL_DIR, 'annotations_cache')
        aps          = []

        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.VOC_YEAR) < 2010 else False
        # print (' - VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        #if not os.path.isdir(self.EVAL_OUTPUTDIR_PKL):
        #    os.mkdir(self.EVAL_OUTPUTDIR_PKL)

        finalMAP = []
        # with tqdm.tqdm_notebook(total = len(self.VOC_CLASSES_)) as pbar:
        for i, cls in enumerate(self.VOC_CLASSES_):
            # pbar.update(1)
            if cls == '__background__':
                continue
            
            rec, prec, ap = self.voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            finalMAP.append([cls, ap])
            # print('AP for {} = {:.4f}'.format(cls, ap))

            #with open(os.path.join(self.EVAL_OUTPUTDIR_PKL, cls + '_pr.pkl'), 'wb') as f:
            #    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        df_MAP = pd.DataFrame(finalMAP, columns=['class', 'mAP'])        
        print('~~~~~~~~')
        print (df_MAP)
        print('~~~~~~~~')
        print('Mean AP = {:.4f}'.format(np.mean(aps)))

        # print('~~~~~~~~')
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        # print('Results computed with the **unofficial** Python eval code.')
        # print('Results should be very close to the official MATLAB eval code.')
        # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        # print('-- Thanks, The Management')
        # print('--------------------------------------------------------------')




class YOLOv2Test():

    def __init__(self):
        pass

class YOLOv1Test():
    
    def __init__(self, model, model_chkp='', IMAGE_SIZE=448, IMAGE_GRID=7):
        if model_chkp != '':
            self.model = self.loadModelChkp(model, model_chkp)
        else:
            self.model = model
        
        if USE_GPU:
            model.cuda()

        self.IMAGE_SIZE = IMAGE_SIZE
        self.IMAGE_GRID = IMAGE_GRID

        self.VOC_CLASSES = (    # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        
        self.VOC_CLASSES_COLOR = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], 
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], 
            [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]]

    def loadModelChkp(self, model, model_chkp):
        if os.path.exists(model_chkp):
            print ('  -- [TEST] Loading Chkpoint : ', model_chkp)
            checkpoint  = torch.load(model_chkp)
            epoch_start = checkpoint['epoch']
            print ('  -- [TEST] Start Epoch : ', epoch_start)
            print ('  -- [TEST][Loss] Train : ', checkpoint['loss_train'])
            print ('  -- [TEST][Loss] Val   : ', checkpoint['loss_val'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print ('')

    def test_decoder_nms(self, bboxes,scores,threshold=0.5):
        '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
        '''
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1) * (y2-y1)

        _,order = scores.sort(0,descending=True)
        keep = []
        while order.numel() > 0:
            try:
                if order.numel() > 1:
                    i = order[0]
                elif order.numel() == 1:
                    i = order.item()

                keep.append(i)

                if order.numel() == 1:
                    break

                xx1 = x1[order[1:]].clamp(min=x1[i])
                yy1 = y1[order[1:]].clamp(min=y1[i])
                xx2 = x2[order[1:]].clamp(max=x2[i])    
                yy2 = y2[order[1:]].clamp(max=y2[i])

                w = (xx2-xx1).clamp(min=0)
                h = (yy2-yy1).clamp(min=0)
                inter = w*h

                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                ids = (ovr<=threshold).nonzero().squeeze()
                if ids.numel() == 0:
                    break
                order = order[ids+1]
            except:
                traceback.print_exc()
                print (' ------------ [ERR!] order : ', order)
                import sys; sys.exit(1)

        return torch.LongTensor(keep)
    
    def test_decoder(self, Y):
        '''
            Input  : Y (tensor) : [1 x 7 x 7 x 30]
            return : (tensor) box[[x1,y1,x2,y2]] label[...]
        '''
        res_boxes      = []
        res_cls_indexs = []
        res_probs      = []
        CELL_SIZE      = 1./self.IMAGE_GRID

        Y        = Y.data
        Y        = Y.squeeze(0) #7x7x30
        contain1 = Y[:,:,4].unsqueeze(2)
        contain2 = Y[:,:,9].unsqueeze(2)
        contain  = torch.cat((contain1,contain2),2)
        mask1    = contain > 0.1
        mask2    = (contain == contain.max()) # we always select the best contain_prob what ever it>0.9
        mask     = (mask1 + mask2).gt(0)
        # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
        for i in range(self.IMAGE_GRID):
            for j in range(self.IMAGE_GRID):
                for b in range(2):
                    # index = min_index[i,j]
                    # mask[i,j,index] = 0
                    if mask[i,j,b] == 1:
                        #print(i,j,b)
                        box          = Y[i,j,b*5:b*5+4]
                        contain_prob = torch.FloatTensor([Y[i,j,b*5+4]])
                        xy           = torch.FloatTensor([j,i]) * CELL_SIZE # cell左上角  up left of cell
                        box[:2]      = box[:2] * CELL_SIZE + xy               # return cxcy relative to image
                        box_xy       = torch.FloatTensor(box.size())        # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                        box_xy[:2]   = box[:2] - 0.5*box[2:]
                        box_xy[2:]   = box[:2] + 0.5*box[2:]
                        max_prob,cls_index = torch.max(Y[i,j,10:],0)
                        if float((contain_prob * max_prob)[0]) > 0.1:
                            res_boxes.append(box_xy.view(1,4))
                            res_cls_indexs.append(cls_index)
                            res_probs.append(contain_prob*max_prob)

        if len(res_boxes) ==0:
            res_boxes      = torch.zeros((1,4))
            res_probs      = torch.zeros(1)
            res_cls_indexs = torch.zeros(1)
        else:
            res_boxes      = torch.cat(res_boxes, dim=0)      #(n,4)
            res_probs      = torch.cat(res_probs, dim=0)      #(n,)
            res_cls_indexs = torch.stack(res_cls_indexs, dim=0) #(n,)
        
        
        keep = self.test_decoder_nms(res_boxes,res_probs)
        return res_boxes[keep], res_cls_indexs[keep], res_probs[keep]

    def test_plot(self, axarr, X, yHat):
        boxes, cls_indexs, probs = yHat
        
        if USE_GPU : img = X.cpu().data.numpy().transpose(1,2,0) 
        else       : img = X.data.numpy().transpose(1,2,0)
        img    = img*255; 
        img    = img.astype(np.uint8)
        h,w,_  = img.shape
        img    = img.copy()
        # print ('  -- img : ', img.dtype, ' || Contiguous : ', img.flags['C_CONTIGUOUS'])

        result = []
        for i,box in enumerate(boxes):
            x1        = int(box[0]*w)
            x2        = int(box[2]*w)
            y1        = int(box[1]*h)
            y2        = int(box[3]*h)
            cls_index = cls_indexs[i]
            cls_index = int(cls_index) # convert LongTensor to int
            prob      = probs[i]
            prob      = float(prob)
            result.append([(x1,y1),(x2,y2), self.VOC_CLASSES[cls_index], '', prob])

        for left_up, right_bottom, class_name, _, prob in result:
            color               = self.VOC_CLASSES_COLOR[self.VOC_CLASSES.index(class_name)]
            label               = class_name + str(round(prob,2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1                  = (left_up[0], left_up[1]- text_size[1])
            # import pdb; pdb.set_trace()
            cv2.rectangle(img, left_up, right_bottom, color, 2)                
            cv2.rectangle(img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            # cv2.putText(img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
            cv2.putText(img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
            
        
        # axarr.imshow(img)
        cv2_imshow(img)
        # cv2.imshow('image',img)

    def test(self, X, Y, plot=0):
        X = Variable(X)
        if USE_GPU:
            X = X.cuda()
        yHat = self.model(X) # [N x 7 x 7 x 30]
        yHat = yHat.cpu()

        boxes_pred       = []
        cls_indexs_pred  = []
        probs_pred       = []

        if plot : 
            f,axarr = plt.subplots(len(yHat),2, figsize=(10,10))
        
        for i_batch, yHat_ in enumerate(yHat):
            # print ('  -- yHat_ : ', yHat_.shape)
            boxes_pred_, cls_indexs_pred_, probs_pred_ = self.test_decoder(yHat_)
            boxes_pred.append(boxes_pred_)
            cls_indexs_pred.append(cls_indexs_pred_)
            probs_pred.append(probs_pred_)
            if (plot):
                self.test_plot(axarr[i_batch][0], X[i_batch], [boxes_pred_, cls_indexs_pred_, probs_pred_])
                print (' - Y : ', Y.shape, ' || ', Y[i_batch].shape)
                boxes_true_, cls_indexs_true_, probs_true_ = self.test_decoder(Y[i_batch])
                self.test_plot(axarr[i_batch][1], X[i_batch], [boxes_true_, cls_indexs_true_, probs_true_])

        return [boxes_pred, cls_indexs_pred, probs_pred]

    def test_metrics_vocap(self, rec,prec,use_07_metric=False):
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0.,1.1,0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec>=t])
                ap = ap + p/11.
        else:
            # correct ap caculation
            mrec = np.concatenate(([0.],rec,[1.]))
            mpre = np.concatenate(([0.],prec,[0.]))

            for i in range(mpre.size -1, 0, -1):
                mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]

            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def getmAP(self, file_annotations, YoloDatasetTest, BATCH_SIZE=16):
        target =  defaultdict(list)
        preds  = defaultdict(list)

        if (1):
            file_list = []
            for line in open(file_annotations).readlines():
                file_list.append(line.strip().split())

            for index,image_file in enumerate(file_list):
                # image_id = image_file[0]
                # image_list.append(image_id)
                # image_list.append(index)
                num_obj = (len(image_file) - 1) // 5
                for i in range(num_obj):
                    x1 = int(image_file[1+5*i])
                    y1 = int(image_file[2+5*i])
                    x2 = int(image_file[3+5*i])
                    y2 = int(image_file[4+5*i])
                    c = int(image_file[5+5*i])
                    class_name = self.VOC_CLASSES[c]
                    target[(index,class_name)].append([x1,y1,x2,y2])
                    if index == 0:
                        print (';')
                        

        if (1):
            image_id = 0
            DataLoaderTest = DataLoader(YoloDatasetTest, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)
            with tqdm.tqdm_notebook(total = len(DataLoaderTest)*BATCH_SIZE) as pbar:
                for i,(X,Y) in enumerate(DataLoaderTest):
                    pbar.update(BATCH_SIZE)
                    yHat = testObj.test(X, Y, plot=False)                
                    [boxes_pred_batch, cls_indexs_pred_batch, probs_pred_batch] = yHat

                    for i_batch, boxes_pred_img in enumerate(boxes_pred_batch):
                        w = 448; h = 448;
                        result = []
                        for i,box in enumerate(boxes_pred_img):
                            x1        = int(box[0]*w)
                            x2        = int(box[2]*w)
                            y1        = int(box[1]*h)
                            y2        = int(box[3]*h)
                            cls_index = cls_indexs_pred_batch[i_batch][i]
                            cls_index = int(cls_index) # convert LongTensor to int
                            prob      = probs_pred_batch[i_batch][i]
                            prob      = float(prob)
                            class_name = self.VOC_CLASSES[cls_index]
                            preds[class_name].append([image_id, prob, x1,y1,x2,y2])

                        image_id += 1

        self.test_metrics(preds, target)                

    def test_metrics(self, preds, target, threshold=0.5, use_07_metric=False,):

        '''
        preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
        target {(image_id,class):[[],]}
        '''
        aps = []
        for i,class_ in enumerate(self.VOC_CLASSES):
            pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
            if len(pred) == 0: # (If there is no abnormality detected in this category)
                ap = -1
                print('---class {} ap {}---'.format(class_,ap))
                aps += [ap]
                break
            #print(pred)
            image_ids  = [x[0] for x in pred]
            confidence = np.array([float(x[1]) for x in pred])
            BB         = np.array([x[2:] for x in pred])
            
            # sort by confidence
            sorted_ind    = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB            = BB[sorted_ind, :]
            image_ids     = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            npos = 0.
            for (key1,key2) in target:
                if key2 == class_:
                    npos += len(target[(key1,key2)]) #Statistics of positive samples of this category, statistics will not be missed here.
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d,image_id in enumerate(image_ids):
                bb = BB[d] #预测框
                if (image_id,class_) in target:
                    BBGT = target[(image_id,class_)] #[[],]
                    for bbgt in BBGT:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(bbgt[0], bb[0])
                        iymin = np.maximum(bbgt[1], bb[1])
                        ixmax = np.minimum(bbgt[2], bb[2])
                        iymax = np.minimum(bbgt[3], bb[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                        if union == 0:
                            print(bb,bbgt)
                        
                        overlaps = inters/union
                        if overlaps > threshold:
                            tp[d] = 1
                            BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                            if len(BBGT) == 0:
                                del target[(image_id,class_)] #删除没有box的键值
                            break
                    fp[d] = 1-tp[d]
                else:
                    fp[d] = 1
            fp   = np.cumsum(fp)
            tp   = np.cumsum(tp)
            rec  = tp/float(npos)
            prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
            #print(rec,prec)
            
            ap = self.test_metrics_vocap(rec, prec, use_07_metric)
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]

        mAP = np.mean(aps)
        print('---map {}---'.format(mAP))
        return  mAP


if __name__ == "__main__":