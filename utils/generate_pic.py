"""
Created on Wed Oct 21 21:10:24 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import h5py
import torch.utils.data as Data

import extract_samll_cubic

def load_dataset(Dataset):
    if Dataset == 'PC':
        uPavia = sio.loadmat('../datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('../datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.999
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
        
    if Dataset == 'DFC2013':
        HS = sio.loadmat('../datasets/DFC2013_Houston.mat')
        gt_HS = sio.loadmat('../datasets/DFC2013_Houston_gt.mat')
        data_hsi = HS['DFC2013_Houston']
        gt_hsi = gt_HS['DFC2013_Houston_gt']
        TOTAL_SIZE = 15029
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    
    if Dataset == 'DN':
        DN = sio.loadmat('../datasets/Dioni.mat')
        gt_DN = sio.loadmat('../datasets/Dioni_gt.mat')
        data_hsi = DN['Dioni']
        gt_hsi = gt_DN['Dioni_gt']
        TOTAL_SIZE = 20024
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'DFC2018':
        DFC = sio.loadmat('../datasets/DFC2018_Houston.mat')
        gt_DFC = sio.loadmat('../datasets/DFC2018_Houston_gt.mat')
        data_hsi = DF