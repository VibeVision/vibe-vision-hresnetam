
"""
Created on Wed Oct 21 21:10:24 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import time
import torch
import numpy as np
import sys
sys.path.append('./utils/')
import d2lzh_pytorch as d2l

def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter: