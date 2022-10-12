import os

import torch

from dataset import LD2011_2014_summary_by_day
import matplotlib.pyplot as plt
import numpy as np
import random

from helper import is_muted
from helper.log import date_tag

FIGURE_DIRECTORY = r"figure"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
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


def plot_forecasting_random_samples(model, dataset, factor, row=2, col=3, filename=None):
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

    if is_muted and filename is not None:
        folder = os.path.exists(os.path.join(FIGURE_DIRECTORY))
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os.path.join(FIGURE_DIRECTORY))
        plt.savefig(os.path.join(FIGURE_DIRECTORY, "{}-Date({}).png".format(filename, date_tag)), dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_ld2011_2014_summary_means_distribution()
