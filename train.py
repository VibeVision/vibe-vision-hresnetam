
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
            test_l_sum, test_num = 0, 0
            X = X.to(device)
            y = y.to(device)
            net.eval() 
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train() 
            n += y.shape[0]
    return [acc_sum / n, test_l_sum]

def train(net, train_iter, valida_iter, loss, optimizer, device, epochs=30, early_stopping=True,
          early_num=20):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()