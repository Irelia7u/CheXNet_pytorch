import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils import data
from PIL import Image

class DataSetDefine(data.Dataset):
    def __init__(self, img_list, label_list, bbox_list, transforms, config):
        self.img_list = img_list
        self.label_tensor_list = label_list
        self.bbox_tensor_list = bbox_list
        self.transforms = transforms
        self.cfg = config

    def __getitem__(self, index):
        img_path_th = self.img_list[index]
        label_th = self.label_tensor_list[index]
        bbox_th = self.bbox_tensor_list[index]
        img_th = Image.open(img_path_th).convert('RGB') 
        img_th = self.transforms(img_th)
        return img_path_th, img_th, label_th, bbox_th
    def __len__(self):
        assert len(self.img_list) == len(self.label_tensor_list)
        assert len(self.img_list) == len(self.bbox_tensor_list)
        return len(self.img_list)


class DataSet(object):
    def __init__(self, config):
        """
        :param config: config parameters
        """
        self.cfg = config

        self.image_list = []
        self.label_tensor_list = []
        self.bbox_tensor_list = []
        
        self.load_list()

        self.transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = DataSetDefine(self.image_list,
                                           self.label_tensor_list,
                                           self.bbox_tensor_list,
                                           self.transforms,
                                           self.cfg)
        self.loader = data.DataLoader(dataset=self.dataset,
                                            batch_size=self.cfg.batch_size,
                                            shuffle=True,
                                            num_workers=self.cfg.num_workers)
      

    def load_list(self):
        info_path = self.cfg.bbox_info

        with open(info_path, 'r') as fp:
            line_list = fp.readlines()

        print('The number of %s samples: %d' %('bbox', len(line_list)))

        for line in line_list:
            items = line.split()
            img_path = os.path.join(self.cfg.data_root, items[0])
            img_label = [int(float(items[1]))]
            img_bbox = items[2:]
            img_bbox = [float(i) for i in img_bbox]
           
 
            self.image_list.append(img_path)
            self.label_tensor_list.append(torch.FloatTensor(img_label))
            self.bbox_tensor_list.append(torch.FloatTensor(img_bbox))
            
if __name__ == '__main__':
    import config
    config.batch_size = 64
    obj = DataSet(config)
    for img, label, bbox in obj.loader:
        print(img.size())
        print(label.size())
        print(bbox)        







