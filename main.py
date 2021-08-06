
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