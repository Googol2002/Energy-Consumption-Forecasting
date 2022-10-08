import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


VARIANCES_FACTOR = 2 * 1e-9
def customized_loss(outputs, labels):
    _means, _variances = outputs[:, :, 0], outputs[:, :, 1]

    return torch.sum((labels - _means) ** 2 / _variances +
                     VARIANCES_FACTOR * _variances)


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.uniform_(layer.bias, 0, 2)


# 十分重要.
# TODO: 或许与具体Task有关，需要更改到Task包下
MEANS_SCALE_FACTOR = 100000
VARIANCES_SCALE_FACTOR = 100000000

class PeriodicalModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, period, means=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2  # Bi-LSTM
        self.batch_size = batch_size
        self.period = period

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)\
            .to(device)
        # 预测均值
        self.mlp_means = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, period)
        ).to(device)
        self.mlp_means.apply(init_weights)    # ReLU在负半轴会失活
        if means is not None:
            self.mlp_means[-1].bias = torch.nn.Parameter(torch.Tensor(means).to(device) / MEANS_SCALE_FACTOR)
        # 预测方差
        self.mlp_variances = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, period)
        ).to(device)
        self.mlp_variances.apply(init_weights)  # ReLU在负半轴会失活
        # self.mlp_variances[-1].bias = torch.nn.Parameter(torch.Tensor(variances).to(device) /
        #                                                  (10 * VARIANCES_SCALE_FACTOR))

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # NOTICE：对于c_0，将其均值初始化在1处是十分必要的！
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device) + 1

        output, (h_n, c_n) = self.lstm(input_seq / MEANS_SCALE_FACTOR, (h_0, c_0))

        return torch.stack((self.mlp_means(torch.cat([h_n[0], h_n[1]], 1)).squeeze() * MEANS_SCALE_FACTOR,
                            self.mlp_variances(torch.cat([h_n[0], h_n[1]], 1)).squeeze() * VARIANCES_SCALE_FACTOR),
                           dim=-1)
