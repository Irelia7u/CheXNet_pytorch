import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.ranking import roc_auc_score



def computeAUC(gt, pred, class_number):
    AUC = []
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    for i in range(class_number):
        AUC.append(roc_auc_score(gt[:,i], pred[:,i]))

    return AUC

def computeIoU(gt, pred):
    GT = np.array([gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]])
    PRED = np.array([pred[0], pred[1], pred[0]+pred[2], pred[1]+pred[3]])
    ixmin = np.maximum(GT[0], PRED[0])
    iymin = np.maximum(GT[1], PRED[1])
    ixmax = np.minimum(GT[2], PRED[2])
    iymax = np.minimum(GT[3], PRED[3])
    iw = np.maximum(ixmax-ixmin, 0)
    ih = np.maximum(iymax-iymin, 0)
    inters = iw * ih
    uni = (gt[2]*gt[3])+(pred[2]*pred[3])-inters
    overlaps = inters/uni
    return overlaps

def computeIoBB(gt, pred):
    GT = np.array([gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]])
    PRED = np.array([pred[0], pred[1], pred[0]+pred[2], pred[1]+pred[3]])
    ixmin = np.maximum(GT[0], PRED[0])
    iymin = np.maximum(GT[1], PRED[1])
    ixmax = np.minimum(GT[2], PRED[2])
    iymax = np.minimum(GT[3], PRED[3])
    iw = np.maximum(ixmax-ixmin, 0)
    ih = np.maximum(iymax-iymin, 0)
    inters = iw * ih
    uni = gt[2]*gt[3]
    overlaps = inters/uni
    return overlaps

def weight_loss(pred, y, weight=None, size_average=True):
    batch_size = pred.size(0)
    weight = weight.unsqueeze(dim=0) 
    neg_weight = weight
    pos_weight = 1-neg_weight
    if torch.cuda.is_available():
        neg_weight = neg_weight.cuda()
        pos_weight = pos_weight.cuda()
    loss = -2 * (y * pos_weight * torch.log(pred) + (1- y) * neg_weight * torch.log(1-pred))
    loss = torch.sum(loss)
    if size_average:
        loss /= batch_size
    return loss  





