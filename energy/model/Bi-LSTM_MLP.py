import numpy as np
import torch
from torch import nn

from energy.dataset import LD2011_2014_summary, construct_dataloader
from energy.log import epoch_log

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 512
LENGTH = 96
EPOCH_STEP = 100  # 超过*次数验证集性能仍未提升，终止
VAL_STEP = 2  # 每经历*次epoch，跑一下验证集
GRADIENT_NORM = 10
WEIGHT_DECAY = 0.01

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
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
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


bias_fn = nn.L1Loss()


def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, bias = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.reshape(-1, LENGTH, 1).to(device)
            y = y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            bias += torch.sum(torch.abs(pred - y) / y).item()

    val_loss /= size
    print(f"Val Error: \n Bias: {(100 * bias / size):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    return val_loss, bias / size


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.reshape(-1, LENGTH, 1).to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # clip
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} Avg loss: {loss / X.shape[0] :>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    dataset = LD2011_2014_summary(length=LENGTH,
                                  csv_file=r"dataset/LD2011_2014.csv",
                                  )
    # transform=lambda t: ((t[0][::4] + t[0][1::4] + t[0][2::4] + t[0][3::4]) / 4, t[1])
    train, val, test = construct_dataloader(dataset, batch_size=BATCH_SIZE)
    print(len(dataset))

    predictor = Bi_LSTM_MPL(input_size=1, hidden_size=256, num_layers=1, output_size=1, batch_size=BATCH_SIZE)

    print('LSTM_MPL model:', predictor)
    loss_function = nn.MSELoss()  # 加速优化
    adam = torch.optim.Adam(predictor.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)

    best_model = None
    min_val_loss = 50000000000
    epoch_step = 0
    val_step = 0
    print('train_sum=', len(train))

    for epoch in range(EPOCH_STEP):
        train_loop(train, predictor, loss_function, adam)
        validation_loss, bias = val_loop(val, predictor, loss_function)
        if validation_loss < min_val_loss:
            min_val_loss = validation_loss
            epoch_log(epoch, "Bi-LSTM_MLP", bias, model=predictor)

