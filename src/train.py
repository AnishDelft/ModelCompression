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

runtime = 'online' # ['local', 'online']

if (runtime == 'online'):
    import src.dataloader as dataloader
    from src.nets2_utils import *
    if (1):
        from src.predict import *
        from src.nets import *

elif runtime == 'local':
    import dataloader as dataloader
    from nets2_utils import *
    from predict import *
    from nets import *


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
                    , BATCH_SIZE, SAVE_INTERNAL
                    , LOGGER='', DEBUG_EPOCHS=-1, verbose=0,pruning_perc=0.,pruning_method="weight"):

        # Step1 - Model Config        
        if (1):
            self.use_cuda = True
            cfgfile       = MODEL_CFG   
            weightfile    = MODEL_WEIGHT
            net_options   = parse_cfg(MODEL_CFG)[0]
            self.model    = Darknet(cfgfile)
            if self.use_cuda:
                self.model = self.model.cuda()
            self.model.load_weights(weightfile)
            # model.print_network()
            
        # Step2 - Dataset
        if (1):
            self.trainlist = PASCAL_TRAIN
            self.testlist  = PASCAL_VALID
            backupdir      = TRAIN_LOGDIR
            nsamples       = file_lines(self.trainlist)
            num_workers   = 4
            self.init_width        = self.model.width
            self.init_height       = self.model.height
            if not os.path.exists(backupdir):
                print (' - backupdir :', backupdir)
                os.mkdir(backupdir)
            kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.use_cuda else {}
        
        # Step3 - Training Params    
        if (1):
            self.batch_size = BATCH_SIZE #int(net_options['batch'])
            max_batches     = int(net_options['max_batches'])
            learning_rate   = float(net_options['learning_rate'])
            momentum        = float(net_options['momentum'])
            decay           = float(net_options['decay'])
            steps           = [float(step) for step in net_options['steps'].split(',')]
            scales          = [float(scale) for scale in net_options['scales'].split(',')]
            max_epochs      = 135
            seed            = int(time.time())
            eps             = 1e-5
            
            torch.manual_seed(seed)
            if self.use_cuda:
                torch.cuda.manual_seed(seed)
            region_loss     = self.model.loss
            region_loss.seen  = self.model.seen
            processed_batches = int(self.model.seen/self.batch_size)
            init_epoch        = int(self.model.seen/nsamples)
        
        # Step3.2 - Optimizer - 
        if (1):
            # params_dict = dict(self.model.named_parameters())
            # params = []
            # for key, value in params_dict.items():
            #     if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            #         params += [{'params': [value], 'weight_decay': 0.0}]
            #     else:
            #         params += [{'params': [value], 'weight_decay': decay*self.batch_size}]
            # optimizer = optim.SGD(self.model.parameters(), 
            #                         lr=learning_rate/self.batch_size, momentum=momentum,
            #                         dampening=0, weight_decay=decay*self.batch_size)
            LR = 0.00001 # 0.00025
            optimizer = optim.SGD(self.model.parameters(), 
                                    lr=LR, momentum=momentum,
                                    dampening=0, weight_decay=decay*self.batch_size)

        # Step4 - Model Saving
        if (1): 
            SAVE_INTERNAL = 10  # epoches
            dot_interval  = 70  # batches

        # Step5 -  Test parameters
        if (1):
            conf_thresh   = 0.25
            nms_thresh    = 0.4
            iou_thresh    = 0.5


        # Step 99 - Random Priting
        if (1):
            print ('')
            print (' -- init_epoch : ', init_epoch)
            print (' -- max_epochs : ', max_epochs)

        if pruning_perc > 0:
            if pruning_method == "filter":
                masks = quick_filter_prune(self.model, pruning_perc)
            else:
                masks = weight_prune(self.model, pruning_perc)
            self.model.set_masks(masks)
            p_rate = prune_rate(self.model,True)
            print(' %s=pruned: %s' % (pruning_method, p_rate))

        for epoch in range(init_epoch, max_epochs):
            
            if (1):
                #lr = self.adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, self.batch_size)
                # lr = 0.00001

                if (1):
                    train_loader = torch.utils.data.DataLoader(
                        dataloader.VOCDatasetv2(PASCAL_TRAIN, shape=(self.init_width, self.init_height),
                                    shuffle=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    train=True,
                                    seen=self.model.seen),
                        batch_size=self.batch_size, shuffle=False, **kwargs)      
                             
            print (' ---------------------------- EPOCH : ', epoch, ' (LR : ',LR,') ---------------------------------- ')

            self.model.train()
            with tqdm.tqdm_notebook(total = len(train_loader)*self.batch_size) as pbar:
                train_loss_total       = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    if (1):
                        if (DEBUG_EPOCHS > -1):
                            if batch_idx > DEBUG_EPOCHS:
                                break

                        pbar.update(self.batch_size)
                        if (batch_idx == 0):
                            print ('  - [INFO] data (or X) : ', data.shape, ' || type : ', data.dtype)       # = torch.Size([1, 3, 416, 416]) torch.float32
                            print ('  - [INFO] target (or Y) : ', target.shape, ' || type : ', target.dtype) # = torch.Size([1, 250]) torch.float64
                            print ('  - [INFO] Total train points : ', len(train_loader), ' || nsamples : ', nsamples)
                    
                    # if (1):
                        #self.adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, self.batch_size)
                        #processed_batches = processed_batches + 1
                    
                    if (1):
                        if self.use_cuda:
                            data   = data.cuda()
                            target = target.float() 
                            # target= target.cuda()
                        data, target = Variable(data), Variable(target)

                    if (1):
                        try:
                            with torch.autograd.detect_anomaly():
                                output           = self.model(data)
                                if (output != output).any():
                                    print ('  -- [DEBUG][train.py] We have some NaNs')
                                    pdb.set_trace()
                                # print ((output != output).any())
                                region_loss.seen = region_loss.seen + data.data.size(0)
                                train_loss       = region_loss(output, target)
                                train_loss_total += train_loss.data
                                    
                                optimizer.zero_grad()
                                train_loss.backward()
                                optimizer.step()

                                if verbose:
                                    print (' - loss : ', train_loss)

                            # for name, param in self.model.named_parameters():
                            #     if param.requires_grad:
                            #         print ('  -- [DEBUG] : ', name, '\t  - \t', round(param.grad.data.sum().item(),3), '   [',param.shape,']')
                        except:
                            traceback.print_exc()
                            pdb.set_trace()

            if pruning_perc > 0:
                print(' pruned: %s' % prune_rate(self.model,False))
                print(' pruned weights consistent after retraining: %s ' % are_masks_consistent(self.model, masks))
                if (epoch + 1) % 5 == 0:
                    logging('save weights to %s/%s-pruned-%s-retrained_%06d.weights' % (backupdir, pruning_method, pruning_perc , epoch+1))
                    self.model.save_weights('%s/%s-pruned-%s-retrained_%06d.weights' % (backupdir, pruning_method, pruning_perc, epoch+1))

            if LOGGER != '':
                train_loss_avg = train_loss_total / len(train_loader)
                print ('   -- train_loss_total : ', train_loss_total, ' || train_loss_avg :', train_loss_avg)
                LOGGER.save_value('Total Loss', 'Train Loss', epoch+1, train_loss_avg)
                train_loss_total       = 0.0

            # logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
            self.model.seen = (epoch + 1) * len(train_loader.dataset)
            # # Save weights
            # if (epoch+1) % SAVE_INTERNAL == 0:
            #     logging('save weights to %s/%s_%06d.weights' % (backupdir, VAL_PREFIX, epoch+1))
            #     self.model.save_weights('%s/%s_%06d.weights' % (backupdir, VAL_PREFIX, epoch+1))

            ## ----------------------- TEST ------------------------
            # self.test(epoch)
            valObj = PASCALVOCEval(self.model, MODEL_CFG, MODEL_WEIGHT, region_loss
                                        , PASCAL_DIR, PASCAL_VALID, VAL_LOGDIR, VAL_PREFIX, VAL_OUTPUTDIR_PKL
                                        , LOGGER, epoch)
            valObj.predict(BATCH_SIZE)
        logging('save weights to %s/%s-pruned-%s-retrained-final_%06d.weights' % (backupdir, pruning_method, pruning_perc , epoch+1))
        self.model.save_weights('%s/%s-pruned-%s-retrained-final_%06d.weights' % (backupdir, pruning_method, pruning_perc, epoch+1))
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

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
    
#     if (torch.cuda.is_available()):
#         if (1):
#             DIR_MAIN         = os.path.abspath('../')
#             print (' - 1. DIR_MAIN :  ', DIR_MAIN)

#         if (1):
#             PASCAL_DIR   = os.path.join(DIR_MAIN, 'data/dataset/VOCdevkit/')
#             PASCAL_TRAIN = os.path.join(DIR_MAIN, 'data/dataset/VOCdevkit/voc_train.txt')
#             PASCAL_VALID = os.path.join(DIR_MAIN, 'data/dataset/VOCdevkit/2007_test.txt')
#             TRAIN_LOGDIR = os.path.join(DIR_MAIN, 'train_data')
#             VAL_LOGDIR   = os.path.join(DIR_MAIN, 'eval_data')
#             VAL_OUTPUTDIR_PKL = os.path.join(DIR_MAIN, 'eval_results')
#             MODEL_CFG    = os.path.join(DIR_MAIN, 'data/cfg/github_pjreddie/yolov2-voc.cfg')
#             MODEL_WEIGHT = os.path.join(DIR_MAIN, 'data/weights/github_pjreddie/yolov2-voc.weights')
#             print (' - 2. MODEL_WEIGHT :  ', MODEL_WEIGHT)

#         if (1):
#             VAL_PREFIX   = 'pretrained'
#             BATCH_SIZE   = 1;
#             print (' - 3. VAL_PREFIX : ', VAL_PREFIX)

#         if (1):
#             LOGGER = ''
#             print (' - 4. Logger : ', LOGGER)


#         if (1):
#             trainObj = YOLOv2Train()
#             trainObj.train(PASCAL_DIR, PASCAL_TRAIN, PASCAL_VALID, TRAIN_LOGDIR, VAL_LOGDIR, VAL_OUTPUTDIR_PKL, VAL_PREFIX
#                         , MODEL_CFG, MODEL_WEIGHT
#                         , BATCH_SIZE
#                         , LOGGER)
#     else:
#         print (' - GPU Issues!!')