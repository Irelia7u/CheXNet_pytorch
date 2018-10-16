import os

num_workers = 4

cam_threshold = 0.9
local_threshold = 0.8
thresh = 0.5
class_number = 14

b1 = 0.9
b2 = 0.999
lr = 0.001
lr_decay_every_epoch = 30
lr_decay_rate = 0.1
patience = 5
weight_decay = 1e-5

threshold = 0.6

epoch = 100
factor = 2

batch_size = int(64/factor)

data_root = '/data1/zhuxin/data/ChestXray/images'
train_info = '/data1/zhuxin/git/chestxray/data/split_by_patient/train.txt'
val_info = '/data1/zhuxin/git/chestxray/data/split_by_patient/val.txt'
test_info = '/data1/zhuxin/git/chestxray/data/split_by_patient/test.txt'
bbox_info = '/data1/zhuxin/data/ChestXray/bbox.txt'
#train_info = '/data1/zhuxin/git/chestxray/data/labels/train_list.txt'
#val_info = '/data1/zhuxin/git/chestxray/data/labels/val_list.txt'
#test_info = '/data1/zhuxin/git/chestxray/data/labels/test_list.txt'
