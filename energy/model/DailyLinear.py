import torch
import torch.nn as nn

from helper import device_manager

class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, x_len, y_len, r_size=24, channels=1, individual=True):
        super(Model, self).__init__()
        self.seq_len = x_len
        self.pred_len = y_len

        self.r_size = r_size
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = channels
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len, device=device_manager.device))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len, device=device_manager.device)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]