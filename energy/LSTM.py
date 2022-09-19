# -*- coding: UTF-8 -*- #
"""
@filename:LSTM.py
@author:201300086
@time:2022-09-19
"""
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from energy.dataset import LD2011_2014_summary,construct_dataloader
import copy
import random
LENGTH=960
EPOCH_STEP=100#超过*次数验证集性能仍未提升，终止
VAL_STEP=2#每经历*次epoch，跑一下验证集
from energy.dataset import BATCH_SIZE

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        #print("batch_size=  ",batch_size,"  seq_len=  ",seq_len)
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        #output, _ = self.lstm(input_seq)
        pred = self.linear(output)
        return pred






Dataset=LD2011_2014_summary(length=LENGTH)
train, val, test = construct_dataloader(Dataset, batch_size=BATCH_SIZE)
Dataset_len=(LD2011_2014_summary.__len__(Dataset))
print(LD2011_2014_summary.__len__(Dataset))
# for i, data in enumerate(train):
# 	# i表示第几个batch， data是batch个X和y（Tensor）
#     print("第 {} 个Batch \n{}".format(i, data))


model=LSTM(input_size=1, hidden_size=64,num_layers= 1,output_size= 1, batch_size=BATCH_SIZE)
print('LSTM model:', model)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training

def get_val_loss(model,val):
    for i, (X,y) in enumerate(val):
        if i <random.randint(1, len(val)):#不会读Dataloader，摆烂
            pass
        else:
            X = X.reshape(LENGTH, BATCH_SIZE, 1)
            y_pred = model(X)
            return loss_function(y_pred, y)

best_model = None
min_val_loss = 50000000000
epoch_step=0
val_step=0
print('train_sum=',len(train))
for epoch ,(X, y) in enumerate(train):
    epoch_step=epoch_step+1
    val_step=val_step+1
    train_loss = []
    print(epoch_step,val_step)
    X=X.reshape(LENGTH,BATCH_SIZE,1)
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    #print(loss)
    train_loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    val_loss = get_val_loss(model, val)
    if val_step>=VAL_STEP:
        val_step=0
        model.eval()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epoch_step=0
            best_model = copy.deepcopy(model)

    if epoch_step>=EPOCH_STEP:
        print('End training for no improvement after {:03d} epoch, train_loss {:.8f} min_val_loss {:.8f}'.format(epoch_step, np.mean(train_loss), min_val_loss))
        break

    print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
    model.train()
