import os
import sys
import tqdm
# dir_main = '~/Work/Netherlands/TUDelft/1_Courses/Sem2/DeepLearning/Project/repo1'
# sys.path.append(dir_main)

import torch

from src.nets2_utils import *
from src.nets import *
from src.dataloader import *

# from nets2_utils import *
# from nets import *
# from dataloader import *

from torch.autograd import Variable
from torchvision import datasets, transforms

torch.cuda.empty_cache()
USE_GPU = torch.cuda.is_available()

def valid(datacfg, cfgfile, weightfile, outfile):
    options      = read_data_cfg(datacfg)
    valid_images = options['valid']
    name_list    = options['names']
    prefix       = 'results'
    names        = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files   = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    model = getYOLOv2(cfgfile, weightfile)
    model.eval()

    valid_dataset = VOCDatasetv2(valid_images, shape=(model.width, model.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    fps = [0]*model.num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(model.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    print (' - Validating : Images : ', len(valid_loader))
    with torch.no_grad():
        with tqdm.tqdm_notebook(total = len(valid_loader)) as pbar:
            for batch_idx, (data, target) in enumerate(valid_loader):
                pbar.update(1)
                data        = data.cuda()
                data        = Variable(data)
                output      = model(data).data
                batch_boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors, 0, 1)
                for i in range(output.size(0)):
                    lineId        = lineId + 1
                    fileId        = os.path.basename(valid_files[lineId]).split('.')[0]
                    width, height = get_image_size(valid_files[lineId])
                    # print(valid_files[lineId])
                    boxes = batch_boxes[i]
                    boxes = nms(boxes, nms_thresh)
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

    for i in range(model.num_classes):
        fps[i].close()

if __name__ == '__main__':
    

    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')

        dir_main = '~/Work/Netherlands/TUDelft/1_Courses/Sem2/DeepLearning/Project/repo1'

        datacfg    = os.path.join(dir_main, 'data/dataset/voc.data')
        cfgfile    = os.path.join(dir_main, 'data/cfg/github_pjreddie/yolov2-voc.cfg')
        weightfile = os.path.join(dir_main, 'data/weights/github_pjreddie/yolov2-voc.weights')
        outfile    = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
