import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 十分重要.
MEANS_SCALE_FACTOR = 100000
VARIANCES_SCALE_FACTOR = 100000000

# def normal_loss(outputs, labels):
#     _means, _variances = outputs[:, :, 0], outputs[:, :, 1]
#
#     return torch.sum((labels - _means) ** 2 / _variances +
#                      VARIANCES_FACTOR * _variances)

def customize_loss(variances_decay):
    def loss(outputs, labels):
        _means, _variances = outputs[:, :, 0], outputs[:, :, 1]

        return torch.sum((labels - _means) ** 2 / _variances +
                         variances_decay * _variances)
    return loss


NORMAL_VARIANCES_FACTOR = 2 * 1e-9
normal_loss = customize_loss(NORMAL_VARIANCES_FACTOR)


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.uniform_(layer.bias, 0, 2)


# 用于为WeeklyModel初始化Mlp
def init_mlp_weights(mlp_sizes, means=None, means_scale_factor=MEANS_SCALE_FACTOR):
    networks = []
    for input_size, output_size in zip(mlp_sizes[: -2], mlp_sizes[1: -1]):
        networks.append(nn.Linear(input_size, output_size))
        networks.append(nn.ReLU())
    outputs = nn.Linear(mlp_sizes[-2], mlp_sizes[-1])
    if means is not None:
        outputs.bias = torch.nn.Parameter(torch.Tensor(means).to(device) / means_scale_factor)
    networks.append(outputs)
    seq = nn.Sequential(*networks).to(device)
    seq.apply(init_weights)

    return seq


class DailyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, period, means=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2  # Bi-LSTM
        self.batch_size = batch_size
        self.period = period

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True) \
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
        self.mlp_means.apply(init_weights)  # ReLU在负半轴会失活
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

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # NOTICE：对于c_0，将其均值初始化在1处是十分必要的！
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device) + 1

        output, (h_n, c_n) = self.lstm(input_seq / MEANS_SCALE_FACTOR, (h_0, c_0))

        return torch.stack((self.mlp_means(torch.cat([h_n[0], h_n[1]], 1)).squeeze() * MEANS_SCALE_FACTOR,
                            self.mlp_variances(torch.cat([h_n[0], h_n[1]], 1)).squeeze() * VARIANCES_SCALE_FACTOR),
                           dim=-1)


CELL_INIT_STD_VARIANCE = 1 / 4

class WeeklyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 batch_size, period, time_size, means=None, mlp_sizes=None,
                 means_scale_factor=MEANS_SCALE_FACTOR,
                 variances_scale_factor=VARIANCES_SCALE_FACTOR):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2  # Bi-LSTM
        self.batch_size = batch_size
        self.period = period
        self.time_size = time_size
        self.means_scale_factor = means_scale_factor
        self.variances_scale_factor = variances_scale_factor

        self.lstm = nn.LSTM(self.input_size + self.time_size, self.hidden_size, self.num_layers,
                            batch_first=True, bidirectional=True).to(device)
        mlp_sizes = mlp_sizes if mlp_sizes else \
            [self.hidden_size * 2 + self.time_size, self.hidden_size, 256, 128, 128, 64, self.period]
        # 预测均值
        self.mlp_means = init_mlp_weights(mlp_sizes, means, self.means_scale_factor)
        # 预测方差
        self.mlp_variances = init_mlp_weights(mlp_sizes)

    def forward(self, energy_xs, time_xs, time_ys):
        batch_size, seq_len = energy_xs.shape[0], energy_xs.shape[1]    # B, L
        predictive_seq_len = time_ys.shape[1]   # L'
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size,
                          self.hidden_size).to(device)
        # NOTICE：对于c_0，将其均值初始化在1处是十分必要的！
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size,
                          self.hidden_size).to(device) * CELL_INIT_STD_VARIANCE + 1

        output, (h_n, c_n) = self.lstm(torch.concat((energy_xs / self.means_scale_factor,
                                                     time_xs), dim=-1), (h_0, c_0))

        # B x L' x 2
        return torch.stack([torch.stack((self.mlp_means(torch.cat([h_n[0], h_n[1], time_ys[:, day]], 1))
                                         * self.means_scale_factor,
                            self.mlp_variances(torch.cat([h_n[0], h_n[1], time_ys[:, day]], 1))
                                         * self.variances_scale_factor),
                            dim=1) for day in range(predictive_seq_len)], dim=1)
