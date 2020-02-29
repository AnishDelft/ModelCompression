import tqdm
import time
import random
import math
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import src.dataloader as dataloader
from src.nets2_utils import *
from src.nets import *
# from src.predict import *


## --------------------------------------- YOLOV2 --------------------------------------- ##
class YOLOv2Train():

    def __init__(self):
        self.model       = ''
        self.optimizer   = ''

        self.trainlist   = ''
        self.testlist    = ''

        self.init_width        = ''
        self.init_height       = ''
        self.batch_size        = ''

    def train(self, PASCAL_DIR, PASCAL_TRAIN, PASCAL_VALID, TRAIN_LOGDIR, VAL_LOGDIR, VAL_OUTPUTDIR_PKL, VAL_PREFIX
                    , MODEL_CFG, MODEL_WEIGHT
                    , BATCH_SIZE
                    , LOGGER=''):
        net_options   = parse_cfg(MODEL_CFG)[0]
        self.trainlist     = PASCAL_TRAIN
        self.testlist      = PASCAL_VALID
        backupdir     = TRAIN_LOGDIR
        nsamples      = file_lines(self.trainlist)
        gpus          = "1"
        ngpus         = 1
        num_workers   = 4
        cfgfile       = MODEL_CFG
        weightfile    = MODEL_WEIGHT

        # model/net configuration
        self.batch_size    = BATCH_SIZE #int(net_options['batch'])
        max_batches   = int(net_options['max_batches'])
        learning_rate = float(net_options['learning_rate'])
        momentum      = float(net_options['momentum'])
        decay         = float(net_options['decay'])
        steps         = [float(step) for step in net_options['steps'].split(',')]
        scales        = [float(scale) for scale in net_options['scales'].split(',')]

        #Train parameters
        max_epochs    = int(max_batches*self.batch_size/nsamples+1)
        self.use_cuda      = True
        seed          = int(time.time())
        eps           = 1e-5
        save_interval = 10  # epoches
        dot_interval  = 70  # batches

        # Test parameters
        conf_thresh   = 0.25
        nms_thresh    = 0.4
        iou_thresh    = 0.5

        if not os.path.exists(backupdir):
            print (' - backupdir :', backupdir)
            os.mkdir(backupdir)

        ###############
        torch.manual_seed(seed)
        if self.use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)

        self.model       = Darknet(cfgfile)
        region_loss      = self.model.loss

        self.model.load_weights(weightfile)
        # model.print_network()

        region_loss.seen  = self.model.seen
        processed_batches = int(self.model.seen/self.batch_size)

        self.init_width        = self.model.width
        self.init_height       = self.model.height
        init_epoch        = int(self.model.seen/nsamples) 

        kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.use_cuda else {}
        

        if self.use_cuda:
            if ngpus > 1:
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()

        params_dict = dict(self.model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                params += [{'params': [value], 'weight_decay': 0.0}]
            else:
                params += [{'params': [value], 'weight_decay': decay*self.batch_size}]
        optimizer = optim.SGD(self.model.parameters(), 
                                lr=learning_rate/self.batch_size, momentum=momentum,
                                dampening=0, weight_decay=decay*self.batch_size)
        
        print ('')
        print (' -- init_epoch : ', init_epoch)
        print (' -- max_epochs : ', max_epochs)

        for epoch in range(init_epoch, max_epochs): 
            
            print (' ---------------------------- EPOCH : ', epoch, ' ---------------------------------- ')
            ## ----------------------- TRAIN ------------------------
            # global processed_batches
            t0 = time.time()
            if ngpus > 1:
                cur_model = self.model.module
            else:
                cur_model = self.model
            
            train_loader = torch.utils.data.DataLoader(
                dataloader.VOCDatasetv2(self.trainlist, shape=(self.init_width, self.init_height),
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            train=True,
                            seen=cur_model.seen,
                            batch_size=self.batch_size),
                batch_size=self.batch_size, shuffle=False, **kwargs)               

            lr = self.adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, self.batch_size)
            # logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
            self.model.train()
            t1 = time.time()
            avg_time = torch.zeros(9)

            train_loss_total       = 0.0
            with tqdm.tqdm_notebook(total = len(train_loader)*self.batch_size) as pbar:
            # with tqdm.tqdm(total = len(train_loader)) as pbar:
                for batch_idx, (data, target) in enumerate(train_loader):

                    if (batch_idx > 20):
                        break

                    pbar.update(self.batch_size)
                    if (batch_idx == 0):
                        print (' - data (or X) : ', data.shape, data.dtype)
                        print (' - target (or Y) : ', target.shape, target.dtype)
                        print (' - Total train points : ', len(train_loader), ' || nsamples : ', nsamples)
                    
                    t2 = time.time()
                    self.adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, self.batch_size)
                    processed_batches = processed_batches + 1
                    
                    if self.use_cuda:
                        data = data.cuda()
                        # target= target.cuda()
                    data, target = Variable(data), Variable(target)
                    optimizer.zero_grad()
                    
                    output = self.model(data)
                    
                    region_loss.seen = region_loss.seen + data.data.size(0)
                    # print (' - region_loss.seen : ', region_loss.seen)
                    print (' - [DEBUG] output : ', output.shape)
                    print (' - [DEBUG] target : ', target.shape)
                    loss = region_loss(output, target)
                    train_loss_total += loss.data;
                    print (' - loss : ', loss, ' || loss.data : ', loss.data)
                    
                    loss.backward()
                    optimizer.step()


            if LOGGER != '':
                train_loss_avg = train_loss_total / len(train_loader)
                print ('   -- train_loss_total : ', train_loss_total, ' || train_loss_avg :', train_loss_avg)
                LOGGER.save_value('Total Loss', 'Train Loss', epoch+1, train_loss_avg)
            t1 = time.time()
            # logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
            # if (epoch+1) % save_interval == 0:
            logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
            cur_model.seen = (epoch + 1) * len(train_loader.dataset)
            cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))

            ## ----------------------- TEST ------------------------
            # self.test(epoch)
            print (' -- Loss : ', self.model.loss)
            valObj = PASCALVOCEval(self.model, MODEL_CFG, MODEL_WEIGHT
                                        , PASCAL_DIR, PASCAL_VALID, VAL_LOGDIR, VAL_PREFIX, VAL_OUTPUTDIR_PKL
                                        , LOGGER, epoch)
            valObj.predict(BATCH_SIZE)
        # end for epoch

    def adjust_learning_rate(self, optimizer, batch, learning_rate, steps, scales, batch_size):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = learning_rate
        for i in range(len(steps)):
            scale = scales[i] if i < len(scales) else 1
            if batch >= steps[i]:
                lr = lr * scale
                if batch == steps[i]:
                    break
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr/batch_size
        return lr

    def test(self, epoch):
        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i

        self.model.eval()
        cur_model = self.model
        num_classes = cur_model.num_classes
        anchors     = cur_model.anchors
        num_anchors = cur_model.num_anchors
        total       = 0.0
        proposals   = 0.0
        correct     = 0.0

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.use_cuda else {}
        test_loader = torch.utils.data.DataLoader(
            dataloader.VOCDatasetv2(self.testlist, shape=(self.init_width, self.init_height),
                        shuffle=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]), train=False),
            batch_size=self.batch_size, shuffle=False, **kwargs)

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output = self.model(data).data
            all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                truths = target[i].view(-1, 5)
                num_gts = truths_length(truths)
        
                total = total + num_gts
        
                for i in range(len(boxes)):
                    if boxes[i][4] > conf_thresh:
                        proposals = proposals+1

                for i in range(num_gts):
                    box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                    best_iou = 0
                    best_j = -1
                    for j in range(len(boxes)):
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou
                    if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                        correct = correct+1
        
        precision = 1.0*correct/(proposals+eps)
        recall = 1.0*correct/(total+eps)
        fscore = 2.0*precision*recall/(precision+recall+eps)
        logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))


## --------------------------------------- YOLOV1 --------------------------------------- ##

class YOLOv1Train():
	
    def __init__(self):
        self.model       = ''
        self.optimizer   = ''

    def train(self, model, criterion, optimizer
				, DataLoaderTrain, DataLoaderTest
				, LEARNING_RATE, EPOCHS, BATCH_SIZE
				, USE_GPU, LOGGER
                , CHKP_LOAD, CHKP_DIR, CHKP_NAME, CHKP_EPOCHS
                , DEBUG):

        if USE_GPU:
            model.cuda ()
		
		# different learning rate
        params      = []
        params_dict = dict(model.named_parameters())
        for key,value in params_dict.items():
            if key.startswith('features'):
                params += [{'params':[value],'lr':LEARNING_RATE*1}]
            else:
                params += [{'params':[value],'lr':LEARNING_RATE}]
        
        if (optimizer == 'SGD'):
            optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        
        print ('')
        epoch_start = 0
        if (CHKP_LOAD):
            path_model = os.path.join(CHKP_DIR, CHKP_NAME)
            if os.path.exists(path_model):
                print ('  -- [TRAIN] Loading Chkpoint : ', path_model)
                checkpoint  = torch.load(path_model)
                epoch_start = checkpoint['epoch']
                print ('  -- [TRAIN] Start Epoch : ', epoch_start)
                print ('  -- [TRAIN][Loss] Train : ', checkpoint['loss_train'])
                print ('  -- [TRAIN][Loss] Val   : ', checkpoint['loss_val'])
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print ('')

        model.train()
        for epoch in range(epoch_start,EPOCHS):
            print ('')
            print (' --------------------------------------------------------- ')
            if epoch >= 30:
                LEARNING_RATE = 0.0001
            if epoch >= 40:
                LEARNING_RATE = 0.00001
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            
            print ('  -- [TRAIN] Epoch % d / % d '  % (epoch +  1 , EPOCHS))
            print ('  -- [TRAIN] LR : {}'.format(LEARNING_RATE))

            
            ## ----------------------- TRAIN ------------------------
            train_loss_total       = 0.0
            train_loss_loc_total   = 0.0
            train_loss_class_total = 0.0
            with tqdm.tqdm_notebook(total = len(DataLoaderTrain)*BATCH_SIZE) as pbar:
                for i,(images,target) in enumerate(DataLoaderTrain):
                    pbar.update(BATCH_SIZE)
                    images = Variable(images)
                    target = Variable(target)
                    if USE_GPU:
                        images,target = images.cuda(),target.cuda()
                    
                    pred                           = model(images)
                    loss_tot, loss_loc, loss_class = criterion(pred,target)
                    train_loss_total               += loss_tot.data
                    train_loss_loc_total           += loss_loc.data
                    train_loss_class_total	       += loss_class.data 	

                    optimizer.zero_grad()
                    loss_tot.backward()
                    optimizer.step()

                    if (DEBUG):
                        break
                
                train_loss_total       /= len(DataLoaderTrain)
                train_loss_loc_total   /= len(DataLoaderTrain)
                train_loss_class_total /= len(DataLoaderTrain)
                if LOGGER != '':
                    LOGGER.save_value('Total Loss', 'Train Loss', epoch, train_loss_total)
                    LOGGER.save_value('Location Loss', 'Train Loc Loss', epoch, train_loss_loc_total)
                    LOGGER.save_value('Class Loss', 'Train Class Loss', epoch, train_loss_class_total)
                print ('  -- [TRAIN] Train Loss : ', train_loss_total)

            
            ## ----------------------- VALIDATION ------------------------
            val_loss_total       = 0.0
            val_loss_loc_total   = 0.0
            val_loss_class_total = 0.0
            model.eval() # to set dropout and batch normalization layers to evaluation mode 
            with torch.no_grad():
                with tqdm.tqdm_notebook(total = len(DataLoaderTest)*BATCH_SIZE) as pbar:
                    for i,(images,target) in enumerate(DataLoaderTest):
                        pbar.update(BATCH_SIZE)
                        images = Variable(images)
                        target = Variable(target)
                        if USE_GPU:
                            images,target = images.cuda(),target.cuda()

                        pred                           = model(images)
                        loss_tot, loss_loc, loss_class = criterion(pred,target)
                        val_loss_total               += loss_tot.data
                        val_loss_loc_total           += loss_loc.data
                        val_loss_class_total	     += loss_class.data 	

                        if (DEBUG):
                            break

                    val_loss_total       /= len(DataLoaderTrain)
                    val_loss_loc_total   /= len(DataLoaderTrain)
                    val_loss_class_total /= len(DataLoaderTrain)
                    if LOGGER != '':
                        LOGGER.save_value('Total Loss', 'Val Loss', epoch, val_loss_total)
                        LOGGER.save_value('Location Loss', 'Val Loc Loss', epoch, val_loss_loc_total)
                        LOGGER.save_value('Class Loss', 'Val Class Loss', epoch, val_loss_class_total)
                    print ('  -- [TRAIN] Validation Loss : ', val_loss_total)

            if USE_GPU:
                print ('  -- [TRAIN] GPU Memory : ', torch.cuda.max_memory_allocated(device=None)/1024/1024/1024, ' GB')
                torch.cuda.reset_max_memory_allocated(device=None)

            ## ----------------------- SAVING ------------------------ 
            if (epoch+1) % CHKP_EPOCHS == 0:
                if not os.path.exists(CHKP_DIR):
                    os.mkdir(CHKP_DIR)

                CHKP_NAME_ = str(CHKP_NAME)
                CHKP_NAME_ = CHKP_NAME_.split('_')[0] + '_epoch%.3d.pkl' % (epoch+1)
                torch.save({
                    'epoch'                : epoch + 1,
                    'model_state_dict'     : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss_train'           : train_loss_total,
                    'loss_val'             : val_loss_total
                        }, os.path.join(CHKP_DIR, CHKP_NAME_)
                    )
                        

        if LOGGER != '' : LOGGER.close()
        self.model = model
        self.optimizer = optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class YOLOv1Loss(nn.Module):

    def __init__(self,S,B,l_coord,l_noobj):
        super(YOLOv1Loss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord # for BBox coord loss
        self.l_noobj = l_noobj # for BBox confidence vals
        
        if (0):
            print ('  - [yoloLoss] S : ', self.S)
            print ('  - [yoloLoss] B : ', self.B)
            print ('  - [yoloLoss] l_coord : ', self.l_coord)
            print ('  - [yoloLoss] l_noobj : ', self.l_noobj)

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    
    def forward(self,pred_tensor,target_tensor):
        verbose = 0
        '''
        pred_tensor   : (tensor) size(batchsize, S, S, Bx5+20=30) [x,y,w,h,c]
        target_tensor : (tensor) size(batchsize, S, S, 30)
        '''
        if (1): # get masks for BBoxes in the image
            N        = pred_tensor.size()[0] #batch_size
            coo_mask = target_tensor[:,:,:,4] > 0
            noo_mask = target_tensor[:,:,:,4] == 0
            if (verbose):
                print (' - N : ', N)
                print (' - coo_mask : ', coo_mask.shape)
                print (' - noo_mask : ', noo_mask.shape)
                print (' - coo_mask : ', coo_mask)
                print (' - noo_mask : ', noo_mask)

            coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
            noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        
        if (1):
            coo_target   = target_tensor[coo_mask].view(-1,30)
            box_target   = coo_target[:,:10].contiguous().view(-1,5)
            class_target = coo_target[:,10:]
            
            coo_pred   = pred_tensor[coo_mask].view(-1,30)
            box_pred   = coo_pred[:,:10].contiguous().view(-1,5) # box[x1,y1,w1,h1,c1]
            class_pred = coo_pred[:,10:]                         #    [x2,y2,w2,h2,c2]
            if (verbose):
                print (' - box_target   : ', box_target.shape)
                print (' - box_target   : ', box_target)
                print (' - class_target : ', class_target)
                print (' - box_pred : ', box_pred.shape)
                print (' - box_pred : ', box_pred)
                print (' - class_pred : ', class_pred)
           
        if (1):
            # compute not contain obj loss
            noo_pred           = pred_tensor[noo_mask].view(-1,30)
            noo_target         = target_tensor[noo_mask].view(-1,30)
            noo_pred_mask      = torch.cuda.ByteTensor(noo_pred.size())
            noo_pred_mask.zero_()
            noo_pred_mask[:,4] = 1;
            noo_pred_mask[:,9] = 1
            noo_pred_c         = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
            noo_target_c       = noo_target[noo_pred_mask]
            nooobj_loss        = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        if (1):
            #compute contain obj loss
            
            coo_response_mask     = torch.cuda.ByteTensor(box_target.size())
            if (verbose):
                print (' - box_target : ', box_target)
                print (' -- coo_response_mask : ', coo_response_mask.shape)
                print (' -- coo_response_mask : ', coo_response_mask)
            coo_response_mask.zero_()
            if verbose:
                print (' -- coo_response_mask : ', coo_response_mask)
            coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
            coo_not_response_mask.zero_()
            box_target_iou        = torch.zeros(box_target.size()).cuda()
            # print (' - box_target_iou : ', box_target_iou)
            
            
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            box1             = box_pred[i:i+2]
            box1_xyxy        = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2]  = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]
            
            box2             = box_target[i].view(-1,5)
            box2_xyxy        = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2]  = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]
            
            if (verbose):
                print (' - box1_xyxy[:,:4] : ', box1_xyxy[:,:4])
                print (' - box2_xyxy[:,:4] : ', box2_xyxy[:,:4])
            
            iou               = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index         = max_index.data.cuda()
            
            coo_response_mask[i+max_index]       = 1
            coo_not_response_mask[i+1-max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
            
            
        
        box_target_iou = Variable(box_target_iou).cuda()
        # print (' - box_target_iou : ', box_target_iou)
        # import sys; sys.exit(1)
        
        # 1.response loss (xpred,ypred vs xtrue,ytrue) + (wpred,hpred vs wpred,hpred)
        box_pred_response       = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response     = box_target[coo_response_mask].view(-1,5)
        loc_loss                = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
        contain_loss            = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)

        # 2.not response loss
        box_pred_not_response        = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response      = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4] = 0
        not_contain_loss             = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,size_average=False)
        

        total_loss = (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N
        return total_loss, self.l_coord*loc_loss, class_loss


if __name__ == "__main__":
    pass


"""
 - Config0 - pjreddie
    - EPOCHS=160
    - LEARNING_RATE=10E3 (/10 at 60 / 90 epochs)
    - WEIGHT_DECAY=0.0005 (L2 regularization)
    - MOMENTUM=0.9
    - DATA_AUG=TRUE (random crops, color shifting)
    - BATCH_SIZE = 64

 - Config1 - 
    - EPOCHS=160
    - LEARNING_RATE=10E4 (/10 at 60 / 90 epochs) <-- reduced this
    - WEIGHT_DECAY=0.0005 (L2 regularization)
    - MOMENTUM=0.9
    - DATA_AUG=TRUE (random crops, color shifting)
    - BATCH_SIZE = 64
 
 - Config2 - 
    - EPOCHS=160
    - LEARNING_RATE=10E5 (/10 at 60 / 90 epochs) <-- reduced this
    - WEIGHT_DECAY=0.0005 (L2 regularization)
    - MOMENTUM=0.9
    - DATA_AUG=TRUE (random crops, color shifting)
    - BATCH_SIZE = 64
 
 - Config3 - 
    - EPOCHS=160
    - LEARNING_RATE=10E3 (/10 at 60 / 90 epochs)
    - WEIGHT_DECAY=0.0005 (L2 regularization)
    - MOMENTUM=0.9
    - DATA_AUG=FALSE (random crops, color shifting) <-- changed this
    - BATCH_SIZE = 64

 - 

"""
