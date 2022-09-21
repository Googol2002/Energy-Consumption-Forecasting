import pytest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from energy.dataset import LD2011_2014, LD2011_2014_summary, construct_dataloader

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

    print("数据规模：{}".format(len(dataset)))
    plt.plot(np.asarray(range(0, LENGTH + 1)), valid_distribution)
    plt.show()


def test_construct_dataloader():
    dataset = LD2011_2014_summary(length=LENGTH,
                                  csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv",
                                  size=10)
    train, val, test = construct_dataloader(dataset, batch_size=128)

    for X, y in test:
        assert(X.shape[1] == LENGTH)

    assert(abs((len(train) + len(val) + len(test)) * 128 - len(dataset)) < 3 * 128)
