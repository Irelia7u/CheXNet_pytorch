import torch
import numpy as np
import config as cfg
import torch.nn as nn
from lib.data_loader import DataSet
from tensorboardX import SummaryWriter
from lib.DensenetModels import DenseNet121
from lib.DensenetModels import DenseNet169
from lib.DensenetModels import DenseNet201
from util.metrics import computeAUC
from util.metrics import weight_loss

net = DenseNet121(cfg.class_number, True)

writer = SummaryWriter()

if torch.cuda.is_available():
    net.cuda()

opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr, betas=(cfg.b1, cfg.b2), weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = cfg.lr_decay_rate, patience = cfg.patience, mode = 'min')
bceloss = nn.BCELoss(size_average=True)
dataset = DataSet(cfg)
weight = dataset.ratio

loss_min = 1000000
for epoch_index in range(cfg.epoch):
    net.train()
    torch.set_grad_enabled(True)
    for batch_index, (img_path, img, label) in enumerate(dataset.train_loader):
        n_iter = epoch_index * len(dataset.train_loader) + batch_index 
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        pred = net(img)
        #loss = weight_loss(pred, label, weight, True)
        loss = bceloss(pred, label)
        loss = loss * 14 
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('train_epoch: %d train_batch: %d loss: %f' %(epoch_index+1, batch_index+1, loss[0]))
        writer.add_scalar('data/train_loss', loss[0], n_iter)

    net.eval()
    torch.set_grad_enabled(False)
    loss_val = 0
    pred_all = torch.FloatTensor().cuda()
    gt_all = torch.FloatTensor().cuda()
    for i, (img_path, img, label) in enumerate(dataset.val_loader):
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        pred = net(img)
        #loss = weight_loss(pred, label, weight, True)
        loss = bceloss(pred, label)
        loss = loss * 14 
        loss_val += loss[0]
        pred_all = torch.cat((pred_all, pred), dim=0)
        gt_all = torch.cat((gt_all, label), dim=0)
    loss_val /= len(dataset.val_loader) 
    auc = computeAUC(gt_all, pred_all, cfg.class_number)
    auc_mean = np.array(auc).mean()

    print('val_epoch: %d loss: %f auc:%f' %(epoch_index+1, loss_val, auc_mean))
    writer.add_scalar('data/val_loss', loss_val, epoch_index)
    writer.add_scalar('data/val_auc', auc_mean, epoch_index)


    scheduler.step(loss_val)
    if loss_val < loss_min:
        loss_min = loss_val
        torch.save({'epoch': epoch_index + 1, 'state_dict': net.state_dict(), 'best_loss': loss_min, 'optimizer' : opt.state_dict()}, 'densenet121/m-' + str(epoch_index) + '.pth.tar')
    elif epoch_index <= 30:
        torch.save({'epoch': epoch_index + 1, 'state_dict': net.state_dict(), 'best_loss': loss_min, 'optimizer' : opt.state_dict()}, 'densenet121/m-' + str(epoch_index) + '.pth.tar')





