import torch
import torch.nn as nn

from helper import device_manager

HIDDEN_SIZE = 24

class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, x_len, y_len, period, time_size, channels=1, individual=True):
        super(Model, self).__init__()
        self.x_len = x_len
        self.y_len = y_len
        self.period = period
        self.time_size = time_size

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = channels
        self.individual = individual

        self.embedding = nn.Sequential(
            nn.Linear(period + time_size, HIDDEN_SIZE, bias=True, device=device_manager.device),
            nn.LeakyReLU()
        )

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.x_len * self.period, self.y_len * self.period, device=device_manager.device))
        else:
            self.Linear = nn.Linear(self.x_len * self.period, self.y_len * self.period, device=device_manager.device)

    def forward(self, energy_xs, time_xs, time_ys):
        # energy_xs: [Batch, Input length, Period, (Channel)]
        seq_last = energy_xs[:, -1:, :].detach()
        x = torch.concat((energy_xs - seq_last, time_xs), dim=-1)

        x = torch.unsqueeze(torch.concat([self.embedding(x[:, i]) for i in range(x.shape[1])], dim=1), -1)
        if self.individual:
            output = torch.zeros([x.size(0), self.y_len * self.period, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            y = output
        else:
            y = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = y.reshape(y.shape[0], self.y_len, -1) + seq_last
        return y  # [Batch, Output length, Channel]