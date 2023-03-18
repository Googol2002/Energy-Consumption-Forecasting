import torch
from torch import nn


class Differential(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, energy_x, time_x, time_y):
        batch_size, x_seq, y_seq = energy_x.shape[0], time_x.shape[1], time_y.shape[1]

        dif_x = energy_x.reshape(batch_size, -1)
        dif_x = torch.concat((torch.zeros(batch_size, 1, device=energy_x.get_device()),
                              dif_x[:, 1:] - dif_x[:, :-1]), dim=1)

        dif_y = self.model(dif_x.reshape(batch_size, x_seq, -1), time_x, time_y)
        y = torch.cumsum(dif_y[:, :, 0].reshape(batch_size, -1), dim=-1) + torch.unsqueeze(energy_x[:, -1, -1], 1)
        return torch.concat((y.reshape(batch_size, y_seq, 1, -1), dif_y[:, :, 1:2, :]), dim=2)


class DifferentialDaily(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, energy_x, time_x, time_y):
        dif_x = torch.concat((torch.zeros_like(energy_x[:, 0:1], device=energy_x.get_device()),
                              energy_x[:, 1:] - energy_x[:, :-1]), dim=1)

        dif_y = self.model(dif_x, time_x, time_y)
        means = torch.cumsum(dif_y[:, :, 0], dim=1) + torch.unsqueeze(energy_x[:, -1], 1)
        return torch.concat((torch.unsqueeze(means, 2), dif_y[:, :, 1:2, :]), dim=2)