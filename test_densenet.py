import torch 
import numpy as np
import config as cfg
import torch.nn as nn
import torch.backends.cudnn as cudnn
from lib.data_loader import DataSet
from lib.DensenetModels import DenseNet121
from util.metrics import computeAUC
from lib.densenet_local import densenet121

cudnn.benchmark = True
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

net = DenseNet121(cfg.class_number, True)
#net = densenet121(True)
if torch.cuda.is_available():
    net.cuda()

net_checkpoint = torch.load('models/densenet121/m-29.pth.tar')
net.load_state_dict(net_checkpoint['state_dict'])

dataset = DataSet(cfg)

net.eval()
torch.set_grad_enabled(False)
pred_all = torch.FloatTensor().cuda()
gt_all = torch.FloatTensor().cuda()
for i, (img_path, img, label) in enumerate(dataset.test_loader):
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    pred = net(img)
    pred_all = torch.cat((pred_all, pred), dim=0)
    gt_all = torch.cat((gt_all, label), dim=0)
    
auc = computeAUC(gt_all, pred_all, cfg.class_number)
auc_mean = np.array(auc).mean()
print('AUC_mean:', auc_mean)

for i in range(cfg.class_number):
    print(CLASS_NAMES[i], ' ', auc[i])

