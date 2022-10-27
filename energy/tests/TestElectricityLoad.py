import random

import pytest
import numpy as np
import matplotlib.pyplot as plt

from energy.dataset.electricity_load import LD2011_2014, LD2011_2014_summary, construct_dataloader, LD2011_2014_summary_by_day

plt.rcParams["font.sans-serif"] = ["SimHei"]    # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

LENGTH = 1000


def test_ld2011_2014_access():
    dataset = LD2011_2014(length=LENGTH,
                          csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv",
                          size=10)
    for index, sample in enumerate(dataset):
        x = sample[0]
        y = sample[1]
        # 不能有超过99%的数据为0，不然视为读入了错误的数据集
        assert ((np.where(x, 0, 1).sum() / x.shape[0]) < 0.99)
        assert (x.shape[0] == LENGTH)
        assert ((x == dataset[index][0]).all())
        assert (y == dataset[index][1])


def test_ld2011_2014_distribution():
    dataset = LD2011_2014(length=LENGTH,
                          size=50)

    valid_distribution = np.zeros(LENGTH + 1)

    for index, sample in enumerate(dataset):
        x = sample[0]
        y = sample[1]

        valid_distribution[np.where(x, 1, 0).sum()] = valid_distribution[np.where(x, 1, 0).sum()] + 1

    plt.plot(np.asarray(range(0, LENGTH + 1)), valid_distribution)
    plt.show()


def test_example():
    dataset = LD2011_2014(length=LENGTH,
                          csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv",
                          size=10)
    print(next(iter(dataset)))


def test_ld2011_2014_summary():
    dataset = LD2011_2014_summary(length=LENGTH,
                                  csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv",
                                  size=10)

    valid_distribution = np.zeros(LENGTH + 1)

    for index, sample in enumerate(dataset):
        x = sample[0]
        # 不能有超过99%的数据为0，不然视为读入了错误的数据集
        valid_distribution[np.where(x, 1, 0).sum()] = valid_distribution[np.where(x, 1, 0).sum()] + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.title.set_text("有效数据分布")
    ax2.title.set_text("Sample展示(4天)")

    print("数据规模：{}".format(len(dataset)))

    PLOT_LENGTH = 96 * 4
    ax1.plot(np.asarray(range(0, LENGTH + 1)), valid_distribution)
    ax2.plot(np.asarray(range(0, PLOT_LENGTH)), dataset[1000][0][:PLOT_LENGTH])
    plt.show()


def test_ld2011_2014_summary_by_day():
    dataset = LD2011_2014_summary_by_day(length=4,
                                         csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    print("数据规模：{}".format(len(dataset)))

    index = random.randint(0, len(dataset))
    fig.suptitle("第{}个Sample的四个连续周期".format(index))

    axs[0][0].plot(np.asarray(range(0, 96)), dataset[index][0][0])
    axs[0][1].plot(np.asarray(range(0, 96)), dataset[index][0][1])
    axs[1][0].plot(np.asarray(range(0, 96)), dataset[index][0][2])
    axs[1][1].plot(np.asarray(range(0, 96)), dataset[index][0][3])

    plt.show()


def test_construct_dataloader():
    dataset = LD2011_2014_summary(length=LENGTH,
                                  csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv",
                                  size=10)
    train, val, test = construct_dataloader(dataset, batch_size=128)

    for X, y in test:
        assert(X.shape[1] == LENGTH)

    assert(abs((len(train) + len(val) + len(test)) * 128 - len(dataset)) < 3 * 128)
