import os

import torch
from matplotlib import ticker
from torch.utils.data import DataLoader

from dataset import LD2011_2014_summary_by_day
import matplotlib.pyplot as plt
import numpy as np
import random

from helper import is_muted, LOG_DIRECTORY, training_recoder
from helper.log import date_tag

FIGURE_DIRECTORY = r"figure"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def plot_ld2011_2014_summary_means_distribution():
    dataset = LD2011_2014_summary_by_day(length=4,
                                         csv_file=r"/dataset/LD2011_2014.csv")

    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    fig.tight_layout(pad=5.0)

    expectations, variances = dataset.statistics()

    for i in range(4):
        y = np.asarray([sample[1][i * 24] for sample in dataset])
        axs[i].plot(range(len(y)), y)
        axs[i].title.set_text("第{}个分量".format(i * 24))
        axs[i].set_xlabel("天数")
        axs[i].set_ylabel("用电量")

    plt.show()
    print("期望:")
    print(expectations)
    print("方差")
    print(variances)


def _figure_directory_path(task_id):
    path = os.path.join(LOG_DIRECTORY, task_id, "figure")
    # 判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def _save_fig(task_id, filename=None):
    if not is_muted and filename is not None:
        path = _figure_directory_path(task_id)
        if not os.path.exists(path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)
        plt.savefig(os.path.join(path, "{}-Date({}).png".format(filename, date_tag)), dpi=300)


def plot_forecasting_random_samples_daily(task_id, model, dataset, factor, row=2, col=3, filename=None):
    fig, axs = plt.subplots(row, col, figsize=(col * 6, row * 6))
    fig.tight_layout(pad=5.0)
    indexes = random.sample(range(len(dataset)), row * col)

    for i in range(row):
        for j in range(col):
            index = i * col + j
            x, y = dataset[indexes[index]]

            with torch.no_grad():
                pred = model(torch.unsqueeze(torch.Tensor(x).to(device), 0)).cpu().numpy()
            x, y = x.reshape(-1), y.reshape(-1)
            means_cup = pred[:, 0].reshape(-1)
            variances_cup = pred[:, 1].reshape(-1)

            axs[i][j].plot(range(y.shape[0]), y)
            axs[i][j].plot(range(y.shape[0]), means_cup, color="red")
            axs[i][j].fill_between(range(y.shape[0]), means_cup - factor * np.sqrt(variances_cup),
                                   means_cup + factor * np.sqrt(variances_cup), facecolor='red', alpha=0.3)
            axs[i][j].title.set_text("Val Sample[{}]".format(index))
            axs[i][j].set_xlabel("Time")
            axs[i][j].set_ylabel("Energy Consumption")

    if not is_muted and filename is not None:
        folder = _figure_directory_path(task_id)
        plt.savefig(os.path.join(FIGURE_DIRECTORY, "{}-Date({}).png".format(filename, date_tag)), dpi=300)

    plt.show()


def plot_forecasting_random_samples_weekly(task_id, model, dataset, factor, size=4, filename=None):
    fig, axs = plt.subplots(size, 1, figsize=(16, size * 6))
    fig.tight_layout(pad=5.0)
    display_dataset = DataLoader(dataset, batch_size=size, shuffle=True)

    device = model.device
    batch, (energy_x, energy_y, time_x, time_y) = next(iter(enumerate(display_dataset)))
    with torch.no_grad():
        pred = model(energy_x.to(device, dtype=torch.float32),
                     time_x.to(device, dtype=torch.float32),
                     time_y.to(device, dtype=torch.float32)).cpu().numpy()

    means_cup, variances_cup = pred[:, :, 0].reshape(size, -1), pred[:, :, 1].reshape(size, -1)
    energy_y = energy_y.reshape(size, -1).cpu().numpy()

    for i, (y, m, v) in enumerate(zip(energy_y, means_cup, variances_cup)):
        axs[i].plot(range(y.shape[0]), y)
        axs[i].plot(range(y.shape[0]), m, color="red")
        axs[i].fill_between(range(y.shape[0]), m - factor * np.sqrt(v),
                            m + factor * np.sqrt(v), facecolor='red', alpha=0.3)
        axs[i].title.set_text("Val Sample[{}]".format(i + 1))
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Energy Consumption")

    _save_fig(task_id, filename)
    plt.show()

def plot_forecasting_weekly_for_comparison(task_id, model, dataset, factor, index):
    def _plot(last_week: bool, mean_curve: bool, confidence_curve: bool, transform,
              yticks=None, percentage=True, filename=None):
        plt.figure(figsize=(12, 8))
        if percentage:
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda temp, position: '%1.0f' % temp + '%'))

        plt.plot(range(y.shape[0]), transform(y), label="实际负载")
        if mean_curve:
            plt.plot(range(y.shape[0]), transform(m), color="red", label="CNN + LSTM 残差")
            if not confidence_curve:
                plt.fill_between(range(y.shape[0]), transform(y),
                                 transform(m), facecolor='red', alpha=0.3)
        if confidence_curve:
            plt.fill_between(range(y.shape[0]), transform(m - factor * np.sqrt(v)),
                             transform(m + factor * np.sqrt(v)), facecolor='red', alpha=0.3, label="高置信度区间")
        if last_week:
            plt.plot(range(y.shape[0]), transform(y_last_week), color="deepskyblue", label="前周填充（基准） 残差")
            if not confidence_curve:
                plt.fill_between(range(y.shape[0]), transform(y),
                                 transform(y_last_week), facecolor='deepskyblue', alpha=0.3)
        # plt.title("样本 [{}]".format(index + 1))
        plt.xlabel("时间")
        if yticks is not None:
            plt.yticks(yticks)
        plt.ylabel("电力负载")
        if percentage:
            plt.title("残差图对比")
        else:
            plt.title("单周预测")
        plt.legend()

        if filename:
            _save_fig(task_id, filename)
        plt.show()

    energy_x, energy_y, time_x, time_y = [torch.Tensor(v).unsqueeze(0) for v in dataset[index]]
    device = model.device
    with torch.no_grad():
        pred = model(energy_x.to(device, dtype=torch.float32),
                     time_x.to(device, dtype=torch.float32),
                     time_y.to(device, dtype=torch.float32)).cpu().numpy()

    means_cup, variances_cup = pred[:, :, 0].reshape(-1), pred[:, :, 1].reshape(-1)
    energy_y = energy_y.reshape(-1).cpu().numpy()
    y_last_week, y, m, v = dataset[index][0][-7:].reshape(-1), energy_y, means_cup, variances_cup
    yticks = np.arange(min(y_last_week - y - 2) // 5 * 5, max(y_last_week - y + 2) // 5 * 5 + 1, 30)
    _plot(True, True, True, lambda v: v, percentage=False, filename="Fig1")

    _plot(True, False, False, lambda v: 100 * (v - y) / y, filename="Fig2")
    _plot(False, True, False, lambda v: 100 * (v - y) / y, filename="Fig3")
    _plot(True, True, False, lambda v: 100 * (v - y) / y, filename="Fig4")
    _plot(True, True, True, lambda v: 100 * (v - y) / y, filename="Fig5")


def plot_forecasting_samples_daily(task_id, model, dataset, factor, index, filename=None):
    plt.figure(figsize=(12, 8))

    energy_x, energy_y, time_x, time_y = [torch.Tensor(v).unsqueeze(0) for v in dataset[index]]
    device = model.device
    with torch.no_grad():
        pred = model(energy_x.to(device, dtype=torch.float32),
                     time_x.to(device, dtype=torch.float32),
                     time_y.to(device, dtype=torch.float32)).cpu().numpy()

    means_cup, variances_cup = pred[:, :, 0].reshape(-1), pred[:, :, 1].reshape(-1)
    energy_y = energy_y.reshape(-1).cpu().numpy()
    y, m, v = energy_y[:48], means_cup[:48], variances_cup[:48]

    transform = lambda v: v
    plt.plot(range(y.shape[0]), transform(y), label="实际负载")
    plt.plot(range(y.shape[0]), transform(m), color="red", label="CNN + LSTM 残差")
    plt.fill_between(range(y.shape[0]), transform(m - factor * np.sqrt(v)),
                     transform(m + factor * np.sqrt(v)), facecolor='red', alpha=0.3, label="高置信度区间")
    plt.title("单日预测".format(index + 1))
    plt.xlabel("时间")
    plt.ylabel("电力负载")
    plt.legend()

    if filename:
        _save_fig(task_id, filename)
    plt.show()

def plot_sensitivity_curve_weekly(task_id, model, dataset, tolerance_range=None, filename=None):
    tolerance_range = tolerance_range if tolerance_range else (0, 2)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    size, shape = len(dataset), dataset[0][1].shape

    @np.vectorize
    def evaluate(tolerance):
        within, utilization = 0, 0
        device = model.device
        with torch.no_grad():
            for (energy_x, energy_y, time_x, time_y) in dataloader:
                energy_x, time_x = energy_x.to(device, dtype=torch.float32), time_x.to(device, dtype=torch.float32)
                energy_y, time_y = energy_y.to(device, dtype=torch.float32), time_y.to(device, dtype=torch.float32)

                pred = model(energy_x, time_x, time_y)

                means = pred[:, :, 0]
                variances = pred[:, :, 1]
                within += torch.sum(energy_y <= means + tolerance * torch.sqrt(variances))
                utilization += torch.sum(energy_y / (means + tolerance * torch.sqrt(variances)))

        return 100 * within.cpu() / (size * shape[0] * shape[1]), 100 * utilization.cpu() / (size * shape[0] * shape[1])

    roc = evaluate(np.linspace(tolerance_range[0], tolerance_range[-1], 50))

    plt.plot(roc[0], roc[1])
    plt.grid()
    plt.xlabel("发电量覆盖用电量情况统计(%)")
    plt.ylabel("能源利用率(%)")
    plt.xticks(np.arange((min(roc[0]) // 5) * 5, 101, 5))
    plt.yticks(np.arange((min(roc[1]) // 5) * 5, 101, 5))
    plt.title("敏感度曲线(Sensitivity Curve)")
    _save_fig(task_id, filename=filename)
    plt.show()


CLIP = 10
def plot_training_process(task_id, filename=None):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    train_loss = np.asarray(training_recoder[task_id].train_loss)
    val_loss = np.asarray(training_recoder[task_id].val_loss)
    gradient_norm = np.asarray(training_recoder[task_id].gradient_norm)
    indexes = np.asarray(range(len(train_loss)), dtype=int)

    clip = CLIP if len(indexes) > CLIP * 2 else 0

    axs[0].title.set_text("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].plot(indexes[clip:], train_loss[clip:], label="Loss on Training Set")
    axs[0].plot(indexes[clip:], val_loss[clip:], label="Loss on Validation Set")
    axs[0].legend()

    axs[1].title.set_text("Training Gradient Norm")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Norm")
    axs[1].plot(indexes[clip:], np.log10(gradient_norm[clip:]), label="Log of Norm to the Base 10")
    axs[1].legend()

    _save_fig(task_id, filename)
    plt.show()

def primitive_plot_forecasting_random_samples_weekly(task_id, dataset, size=4, filename=None):
    fig, axs = plt.subplots(size, 1, figsize=(16, size * 6))
    fig.tight_layout(pad=5.0)
    display_dataset = DataLoader(dataset, batch_size=size, shuffle=True)

    batch, (energy_x, energy_y, time_x, time_y) = next(iter(enumerate(display_dataset)))

    
    energy_y = energy_y.reshape(size, -1).numpy()
    energy_x = energy_x.reshape(size, -1).numpy()

    for i, (y, y_pre) in enumerate(zip(energy_y, energy_x)):
        axs[i].plot(range(y.shape[0]), y)
        axs[i].plot(range(y.shape[0]), y_pre, color="red")
        axs[i].title.set_text("Val Sample[{}]".format(i + 1))
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Energy Consumption")

    _save_fig(task_id, filename)
    plt.show()


if __name__ == "__main__":
    plot_ld2011_2014_summary_means_distribution()
    # fig, axs = plt.subplots(size, 1, figsize=(16, size * 6))
    # fig.tight_layout(pad=5.0)