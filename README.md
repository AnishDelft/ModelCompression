# ModelCompression
Weight and filter Pruning based model compression of YOLO v2 object detection network

# Topic - Efficient Deep Learning - Model compression via pruning for object detection

# Work Done
| Week |      Date       |      Topic            |                        Work Done                       |
| ---- | --------------- | ---------------       | -----------------------------------------------------  |  
|   1  |  21/04 - 27/04  |   Research            | Read up on all 3 model-compression types for comparison|
|   2  |  28/04 - 04/05  |   Object Detection    | Read up on SSD  |   
|   3  |  05/05 - 11/05  |   Code Play - Obj Det |   |
|   4  |  12/05 - 18/05  |   Code Play - Comp    |   |

# References
1. [PASCAL VOC 2007 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
2. [YOLO Repo](https://github.com/xiongzihua/pytorch-YOLO-v1/blob/master/dataset.py)

## Models - YOLOv2
### Loss Function
    - 3 loss elements
        - Box Loss
            - position loss  : [ [(x_true - x_predict)^2             + (y_true - y_predict)^2)         ]  * coord_mask ] * coord_scale
            - dimension loss : [ [(sqrt(w_true) - sqrt(w_predict)^2) + (sqrt(h_true - sqrt(h_predict)))]  * coord_mask ] * coord_scale
        - Confidence Loss
            - [ (c_box_true - c_box_pred)^2 ] * conf_mask         * object_scale
            - [ (c_box_true - c_box_pred)^2 ] * [mask_cell_noobj] * noobject_scale
        - Class Loss
            - [(class_true - class_predict)^2] * cls_mask * class_scale
    
    - Params (yolov2-voc.cfg)
        - coord_scale    = 1 
        - object_scale   = 5
        - noobject_scale = 1 
        - class_scale    = 1

        - coord_mask ~ cls_mask
        - conf_mask

## Things to Fix
    - [Done] prevent bloating of loss_w, loss_h 
    - [Done] prevent tw, th from going negative!!!
    - [Done] understand allocation of tx, ty, tw, th (train.build_targets())
    - [Done] Understand predict.py
    - [Done] Understand region_loss within predict.py [No new learning]
    - [Done] Make LR = 0.00001
    - [Done] Understand loss via cls/tcls
    - [Done] Change object_scale = 1
    - [Done] Change conf_mask = torch.zeros 
    - [Done] Prevent Exploding gradients (key-change = torch.sqrt!!)

## How was it fixed
    - src/train.py
        - LR = 0.00001
        - using with `torch.autograd.detect_anomaly():`
        - make (data, target).dtype == float32
    - src/nets.py
        - w,h = torch.exp(output(nB,nA,2,nW,nH))
        - [??] self.noobject_scale = 1, self.object_scale = 5 --> 1 (does it matter ??)
        - conf_mask   = torch.ones(nB, nA, nH, nW) --> conf_mask   = torch.zeros(nB, nA, nH, nW) 
        - build_targets() : if (seen < 12800) - commented this if block
        - 


## Datasets and Evaluation
1. PASCAL VOC
    - VOCDevKit/VOC20{07,12}/JPEGImages/{%6d.jpg}
        - these are images with similiar, but different sizes
    - VOCDevKit/VOC20{07,12}/Annotations/{%6d.xml}
        - [class, xmin, xmax, ymin, ymax]
        - Note : These are pixel coordinates wrt image origin (top-left) and not normalized

3. Code
    - predict.py
        - Saves as [fileId, class_prob, xmin, ymin, xmax, ymax] (for each class_id)
        - Note : These are pixel coordinates wrt image origin (top-left) and not normalized

2. YOLO Arch (notice that now we have [x,y,w,h] instead of [xmin,ymin,xmax,ymax])
    - output = [nB,nA*(5+nC),nW,NH]
        - nB = batch_size
        - nA = anchor_size
        - 5 = [x,y,w,h, box_confidence] 
        - nC = Total Classes 
        - nW = grid cells - x-axis
        - nH = grid cells - y-axis
        - Eg : [1, 5*(5+20), 13, 13]
        - Note : 
            - x = [nB, nA, nH, nW] - Eg : [1,5,13,13]
                - Note : this x is the BBox centre and its coords are wrt a grid-cell-origin (top-left of a grid-cell)
            - y = [nB, nA, nH, nW]
                - Note : this y is the BBox centre and its coords are wrt a grid-cell-origin (top-left of a grid-cell)
            - w = [nB, nA, nH, nW]
            - h = [nB, nA, nH, nW]

    - target = [nB, (5 + nC)*MAX_BOXES]
        - MAX_BOXES = maximum bounding boxes in an image
        - Eg : [1, (5 + 20)*50] = [1,250]
        - Note : 
            - these are extracted from .txt files generated via src/dataloader.py (setup_VOC --> )
            - these convert [xmin,ymin,xmax,ymax] -- [norm_centre_x, norm_centre_y, norm_width, norm_height]
            - here norm means normalized
            - the values are still wrt image origin (top-left)
                - thus they are modified in src/nets.py (RegionLoss)


## Next Steps 
    - Share configurations with everyone and ask them to report the results (mAP) by 9th of June
    - Visualisation 1, bounding box of 5 images: before pruning vs after retraining
    - Visualisation 2, 3D weights plots: before pruning vs after retraining
    - Keep up with the report (Anish and Soroosh)
    - Poster presentation

