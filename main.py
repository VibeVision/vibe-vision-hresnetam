
"""
Created on Fri Oct 23 20:58:38 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
import numpy as np
import sys
sys.path.append('./networks')
import network
import train
sys.path.append('./utils')
from generate_pic import aa_and_each_accuracy, sampling1, sampling2, load_dataset, generate_png, generate_iter
import record, extract_samll_cubic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0')

seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')
global Dataset
dataset = 'PC'
Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset)
print(data_hsi.shape)

ROWS, COLUMNS, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = int(max(gt))
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 2
PATCH_LENGTH = 3
lr, num_epochs, batch_size = 0.0002, 300, 32
loss = torch.nn.CrossEntropyLoss()

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
SAMPLES_NUM = 200
