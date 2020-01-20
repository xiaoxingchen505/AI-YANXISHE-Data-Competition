# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4') 

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

# input dataset
train_jpg = pd.read_csv('./豆腐和土豆/train.csv', names=['id', 'label'])
train_jpg['id'] = train_jpg['id'].apply(lambda x: './豆腐和土豆/train/' + str(x) + '.jpg')
    
class QRDataset(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_df = img_df
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.img_df.iloc[index]['id']).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img,torch.from_numpy(np.array(self.img_df.iloc[index]['label']))
    
    def __len__(self):
        return len(self.img_df)


dataset = QRDataset(train_jpg,
                transforms.Compose([
                            # transforms.RandomGrayscale(),
                            transforms.Resize((512, 512)),
                            transforms.RandomAffine(5),
                            # transforms.ColorJitter(hue=.05, saturation=.05),
                            # transforms.RandomCrop((88, 88)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            # transforms.ToTensor(),
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
                   )