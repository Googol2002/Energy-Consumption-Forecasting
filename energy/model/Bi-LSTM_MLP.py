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
from energy.dataset import LD2011_2014_summary, construct_dataloader
import copy
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 256
LENGTH = 192
EPOCH_STEP = 100  # 超过*次数验证集性能仍未提升，终止
VAL_STEP = 2  # 每经历*次epoch，跑一下验证集

class Bi_LSTM_MPL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2  # Bi-LSTM
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)\
            .to(device)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        ).to(device)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # NOTICE：对于c_0，将其均值初始化在1处是十分必要的！
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device) + 1

        output, _ = self.lstm(input_seq, (h_0, c_0))

        return self.mlp(output[:, -1, :]).squeeze()


def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.reshape(-1, LENGTH, 1).to(device)
            y = y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()

    val_loss /= size
    print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    return val_loss


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.reshape(-1, LENGTH, 1).to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    dataset = LD2011_2014_summary(length=LENGTH,
                                  csv_file=r"../../dataset/LD2011_2014.csv")
    train, val, test = construct_dataloader(dataset)
    print(len(dataset))

    predictor = Bi_LSTM_MPL(input_size=1, hidden_size=128, num_layers=1, output_size=1, batch_size=128)

    print('LSTM_MPL model:', predictor)
    loss_function = nn.L1Loss()  # 加速优化
    adam = torch.optim.Adam(predictor.parameters(), lr=0.001)

    best_model = None
    min_val_loss = 50000000000
    epoch_step = 0
    val_step = 0
    print('train_sum=', len(train))
    for epoch in range(EPOCH_STEP):
        train_loop(train, predictor, loss_function, adam)
        validation_loss = val_loop(val, predictor, loss_function)
        if validation_loss < min_val_loss:
            validation_loss = min_val_loss
            torch.save(predictor.state_dict(), 'best_model.pth')

