import numpy as np
import torch
from torch import nn

from energy.dataset import LD2011_2014_summary, construct_dataloader, LD2011_2014_summary_by_day
from energy.log import epoch_log, log_printf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 16
HIDDEN_SIZE = 512
PERIOD = 96
LENGTH = 30
EPOCH_STEP = 400
GRADIENT_NORM = 10
WEIGHT_DECAY = 0.01


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.uniform_(layer.bias, 0, 2)


# 十分重要.
SCALE_FACTOR = 100000

class Bi_LSTM_MPL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, means):
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
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, PERIOD)
        ).to(device)
        self.mlp.apply(init_weights) # ReLU在负半轴会失活
        self.mlp[-1].bias = torch.nn.Parameter(torch.Tensor(means).to(device) / SCALE_FACTOR)

    net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    net.apply(init_weights)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # NOTICE：对于c_0，将其均值初始化在1处是十分必要的！
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device) + 1

        output, (h_n, c_n) = self.lstm(input_seq / SCALE_FACTOR, (h_0, c_0))

        return self.mlp(torch.cat([h_n[0], h_n[1]], 1)).squeeze() * SCALE_FACTOR


bias_fn = nn.L1Loss()

def check_gradient_norm(model):
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print("Gradient norm: {:>7f}".format(total_norm))


def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, bias = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            bias += torch.sum(torch.abs(pred - y) / y).item()

    val_loss /= (size * PERIOD)
    log_printf("Bi-LSTM_MLP", f"Val Error: \n Bias: {(100 * bias / (size * PERIOD)):>0.3f}%, Avg loss: {val_loss:>8f} \n")

    return val_loss, bias / size


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        check_gradient_norm(model)
        # clip
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} Avg loss: {loss / (X.shape[0] * PERIOD) :>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    dataset = LD2011_2014_summary_by_day(length=LENGTH,
                                         csv_file=r"dataset/LD2011_2014.csv",
                                         )
    train, val, test = construct_dataloader(dataset, batch_size=BATCH_SIZE)
    print(len(dataset))

    expectations, variances = dataset.statistics()

    predictor = Bi_LSTM_MPL(input_size=PERIOD, hidden_size=HIDDEN_SIZE, num_layers=1, output_size=PERIOD,
                            batch_size=BATCH_SIZE, means=expectations)

    print('Bi-LSTM_MLP_period model:', predictor)
    loss_function = nn.MSELoss()    # 加速优化
    adam = torch.optim.Adam(predictor.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)

    best_model = None
    min_val_loss = 50000000000
    epoch_step = 0
    val_step = 0
    print('train_sum=', len(train))

    val_loop(val, predictor, loss_function);
    for epoch in range(EPOCH_STEP):
        train_loop(train, predictor, loss_function, adam)
        validation_loss, bias = val_loop(val, predictor, loss_function)
        if validation_loss < min_val_loss:
            min_val_loss = validation_loss
            # epoch_log(epoch, "Bi-LSTM_MLP", bias, model=predictor)

