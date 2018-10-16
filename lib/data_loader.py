import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils import data
from PIL import Image

class DataSetDefine(data.Dataset):
    def __init__(self, img_list, label_list, transforms, config):
        self.img_list = img_list
        self.label_tensor_list = label_list
        self.transforms = transforms
        self.cfg = config

    def __getitem__(self, index):
        img_path_th = self.img_list[index]
        label_th = self.label_tensor_list[index]
        img_th = Image.open(img_path_th).convert('RGB') 
        img_th = self.transforms(img_th)
        return img_path_th, img_th, label_th
    def __len__(self):
        assert len(self.img_list) == len(self.label_tensor_list)
        return len(self.img_list)


class DataSet(object):
    def __init__(self, config):
        """
        :param config: config parameters
        """
        self.cfg = config

        self.train_image_list = []
        self.train_label_tensor_list = []

        self.val_image_list = []
        self.val_label_tensor_list = []

        self.test_image_list = []
        self.test_label_tensor_list = []
        
        self.load_list(data_type='train')
        self.load_list(data_type='val')
        self.load_list(data_type='test')
     
        self.cal_ratio()

        self.train_transforms = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_dataset = DataSetDefine(self.train_image_list,
                                           self.train_label_tensor_list,
                                           self.train_transforms,
                                           self.cfg)
        self.val_dataset = DataSetDefine(self.val_image_list,
                                         self.val_label_tensor_list,
                                         self.val_transforms,
                                         self.cfg)
        self.test_dataset = DataSetDefine(self.test_image_list,
                                          self.test_label_tensor_list,
                                          self.test_transforms,
                                          self.cfg)
        self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                            batch_size=self.cfg.batch_size,
                                            shuffle=True,
                                            num_workers=self.cfg.num_workers)
        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                          batch_size=self.cfg.batch_size,
                                          shuffle=False,
                                          num_workers=self.cfg.num_workers)
        self.test_loader = data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.cfg.batch_size,
                                           shuffle=False,
                                           num_workers=self.cfg.num_workers)
      

    def load_list(self, data_type):
        assert data_type in ['train', 'val', 'test']
        if data_type == 'train':
            info_path = self.cfg.train_info
        elif data_type == 'val':
            info_path = self.cfg.val_info
        elif data_type == 'test':
            info_path = self.cfg.test_info

        with open(info_path, 'r') as fp:
            line_list = fp.readlines()

        print('The number of %s samples: %d' %(data_type, len(line_list)))

        for line in line_list:
            items = line.split()
            img_path = os.path.join(self.cfg.data_root,items[0])
            img_label = items[1:]
            img_label = [int(float(i)) for i in img_label]
           
 
            if data_type == 'train':
                self.train_image_list.append(img_path)
                self.train_label_tensor_list.append(torch.FloatTensor(img_label))
            elif data_type == 'val':
                self.val_image_list.append(img_path)
                self.val_label_tensor_list.append(torch.FloatTensor(img_label))
            elif data_type == 'test':
                self.test_image_list.append(img_path)
                self.test_label_tensor_list.append(torch.FloatTensor(img_label))
    
    def cal_ratio(self):
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']   
        img_num = len(self.train_label_tensor_list) 
        train_label = torch.stack(self.train_label_tensor_list, dim=0)
        self.ratio = torch.sum(train_label, dim=0)/img_num
        for i in range(14):
            print(CLASS_NAMES[i], ':', float(self.ratio[i]))  

    
if __name__ == '__main__':
    import config
    config.batch_size = 64
    obj = DataSet(config)
    for img_path, img, label in obj.test_loader:
        print(img.size())
        print(label.size())        







