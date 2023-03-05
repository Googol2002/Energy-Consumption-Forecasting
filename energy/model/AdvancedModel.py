import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from helper.device_manager import device

# 十分重要.
MEANS_SCALE_FACTOR = 100000
VARIANCES_SCALE_FACTOR = 100000000

STABILIZING_FACTOR = 1e-5
def customize_loss(variances_decay):
    def loss(outputs, labels):
        _means, _variances = outputs[:, :, 0], outputs[:, :, 1]

        return torch.sum((labels - _means) ** 2 / (_variances + STABILIZING_FACTOR) +
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

CELL_INIT_STD_VARIANCE = 1 / 4

class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 batch_size, period, time_size, kernel_size,
                 means=None, mlp_sizes=None,
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

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=kernel_size, padding="same", padding_mode="reflect"),
            nn.Conv1d(2, 4, kernel_size=kernel_size, padding="same", padding_mode="reflect"),
            nn.MaxPool1d(kernel_size, stride=4, padding=kernel_size // 2)
        ).to(device)

        # 138需要后续计算
        self.lstm = nn.LSTM(self.input_size * 1 + self.time_size, self.hidden_size, self.num_layers,
                            batch_first=True, bidirectional=True).to(device)
        mlp_sizes = mlp_sizes if mlp_sizes else \
            [self.hidden_size * 2 + self.time_size, self.hidden_size, 256, 128, 64, self.period]
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

        shape = energy_xs.shape
        features = self.cnn(energy_xs.reshape(shape[0], 1, -1) / self.means_scale_factor)
        features = features.reshape(shape[0], shape[1], -1)

        output, (h_n, c_n) = self.lstm(torch.concat((features, time_xs), dim=-1), (h_0, c_0))

        # B x L' x 2
        return torch.stack([torch.stack((self.mlp_means(torch.cat([h_n[0], h_n[1], time_ys[:, day]], 1))
                                         * self.means_scale_factor,
                            self.mlp_variances(torch.cat([h_n[0], h_n[1], time_ys[:, day]], 1))
                                         * self.variances_scale_factor),
                            dim=1) for day in range(predictive_seq_len)], dim=1)

# 增加了 固定长度的 Attention 功能
class CNN_Attention_Model(CNNModel):
    def __init__(self, *args, attention_size=30, **kwargs):
        super(CNN_Attention_Model, self).__init__(*args, **kwargs)
        self.attention = nn.Parameter(torch.randn(2, 1, attention_size, 1, device=device))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, energy_xs, time_xs, time_ys):
        batch_size, seq_len = energy_xs.shape[0], energy_xs.shape[1]  # B, L
        predictive_seq_len = time_ys.shape[1]  # L'
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size,
                          self.hidden_size).to(device)
        # NOTICE：对于c_0，将其均值初始化在1处是十分必要的！
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size,
                          self.hidden_size).to(device) * CELL_INIT_STD_VARIANCE + 1

        shape = energy_xs.shape
        features = self.cnn(energy_xs.reshape(shape[0], 1, -1) / self.means_scale_factor)
        features = features.reshape(shape[0], shape[1], -1)

        output, (h_n, c_n) = self.lstm(torch.concat((features, time_xs), dim=-1), (h_0, c_0))
        # TODO: 分别处理正向和反向，直接加权有些不合理
        hidden_state = torch.sum(torch.cat([
            output[:, :, :self.hidden_size] * self.softmax(self.attention[0]),
            output[:, :, self.hidden_size:] * self.softmax(self.attention[1])
        ], 2), dim=1)

        # B x L' x 2
        return torch.stack([torch.stack((self.mlp_means(torch.cat([hidden_state, time_ys[:, day]], 1))
                                         * self.means_scale_factor,
                                         self.mlp_variances(torch.cat([h_n[0], h_n[1], time_ys[:, day]], 1))
                                         * self.variances_scale_factor),
                                        dim=1) for day in range(predictive_seq_len)], dim=1)


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 batch_size, period, time_size, n_head, n_layers, means=None, mlp_sizes=None,
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

        # 禁止了 dropout
        encoder_layers = TransformerEncoderLayer(self.input_size + self.time_size, n_head, hidden_size, dropout=0)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers).to(device)

        mlp_sizes = mlp_sizes if mlp_sizes else \
            [self.hidden_size * 2 + self.time_size, self.hidden_size, 256, 128, 128, 64, self.period]
        # 预测均值
        self.mlp_means = init_mlp_weights(mlp_sizes, means, self.means_scale_factor)
        # 预测方差
        self.mlp_variances = init_mlp_weights(mlp_sizes)

    def forward(self, energy_xs, time_xs, time_ys):
        batch_size, seq_len = energy_xs.shape[0], energy_xs.shape[1]    # B, L
        predictive_seq_len = time_ys.shape[1]   # L'
        # h_0 = torch.randn(self.num_directions * self.num_layers, batch_size,
        #                   self.hidden_size).to(device)
        # # NOTICE：对于c_0，将其均值初始化在1处是十分必要的！
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size,
        #                   self.hidden_size).to(device) * CELL_INIT_STD_VARIANCE + 1

        output = self.transformer_encoder(torch.concat((energy_xs / self.means_scale_factor,
                                          time_xs), dim=-1))

        # B x L' x 2
        return torch.stack([torch.stack((self.mlp_means(torch.cat([output, time_ys[:, day]], 1))
                                         * self.means_scale_factor,
                            self.mlp_variances(torch.cat([output, time_ys[:, day]], 1))
                                         * self.variances_scale_factor),
                            dim=1) for day in range(predictive_seq_len)], dim=1)
