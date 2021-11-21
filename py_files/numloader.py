# These are all the modules we'll use later.

import re
import os
import h5py
import json
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from PIL import Image
import random
from . import utils
from . import transforms as T


# def get_transform(train):
#     transforms = []
#     # converts the image, a PIL image, into a PyTorch Tensor
#     transforms.append(T.ToTensor())
#     if train:
#         # during training, randomly flip the training images
#         # and ground-truth for data augmentation
#         transforms.append(T.RandomHorizontalFlip(0.5))

#     return T.Compose(transforms)
# RESIZE_SIZE = 224
trans = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    # transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.ToTensor(), 
    # transforms.Normalize([0.5], [0.5]),
])


class NumberDataset(Dataset):
    def __init__(self, is_train=False):

        self.is_train = is_train
        # self.transform = get_transform(True)
        self.transform = trans
        # self.size = RESIZE_SIZE
        
        if self.is_train:
            all_info = './dataset/multi_bbox_info.json'
            self.filename_list = os.listdir('./dataset/train')
            self.filename_list = list(filter(lambda x: re.match('[\d+]+[.]+png$', x) != None, self.filename_list))
            # here, we use re since there are some duplicate file
            # and the file will be named after '31055 (1).png'
            # but initial image will be named after like '31055.png' format
            # is the format: 1 or more number + .png
            # json file is normal in the other ipynb file
            # but in this file it will generate some duplicate new data
            with open(all_info, 'r') as f:
                all_info = json.load(f)
                self.train_data = all_info['train_data']
                del all_info

        else:
            self.filename_list = os.listdir('./dataset/test')
            print('Testing set', len(self.filename_list))

    def __getitem__(self, idx):
        if self.is_train:
            data = self.train_data[idx]
            filename = data['filename']
            boxes = data['boxes']
            # a list with many dicts, every dict has 5 items
            # x1, y1, x2, y2, label

            # image
            img = Image.open(f'./dataset/train/{filename}')

            target = self.load_annotations(boxes)
            target["image_id"] = torch.tensor([idx])

            if self.transform is not None:
                img = self.transform(img)
            return img, target

        else:
            # img_id_dict = {}
            path = self.filename_list[idx]
            img = Image.open(f'./dataset/test/{path}')
            if self.transform is not None:
                img = self.transform(img)
            img_id = int(re.match('\d+', path).group(0))
            
            return img, img_id

    def load_annotations(self, boxes, rw=1, rh=1):
        # transform from [label, left, top, width, height] to [x1, y1, x2, y2]
        le = len(boxes)
        box_list = []
        label_list = []
        for i in range(le): 
            x1 = boxes[i]['left']
            y1 = boxes[i]['top']
            x2 = x1 + boxes[i]['width']
            y2 = y1 + boxes[i]['height']

            box_list.append([x1, y1, x2, y2])
            if boxes[i]['label'] == 10:
            # if label=10, that means number is 0
                boxes[i]['label'] = 0.0
            la = boxes[i]['label']
            
            label_list.append(la)
            # if we have x1x2y1y2, we have a box.
        box_list = torch.as_tensor(box_list, dtype=torch.float32)
        label_list = torch.as_tensor(label_list, dtype=torch.int64)

        area = (box_list[:, 3] - box_list[:, 1]) * (box_list[:, 2] - box_list[:, 0])

        iscrowd = torch.zeros((le,), dtype=torch.int64)

        target = {}
        target["boxes"] = box_list
        target["labels"] = label_list
        # target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def num_classes(self):
        return 10
    def __len__(self):
        return len(self.filename_list)
