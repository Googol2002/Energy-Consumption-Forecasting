from math import ceil

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, random_split

import pandas as pd
import numpy as np


# 返回sorted_list中的区间序号
# sorted_list 描述了一些“左闭右开区间”的右边界，返回下表标k iff a_{k} <= x < a_k+1
def search_sorted(sorted_list, x, suggestion=0):
    # 更新为Binary Search
    if not (0 <= x < sorted_list[-1]):
        raise IndexError()
    if sorted_list[suggestion] <= x < sorted_list[suggestion + 1]:
        return suggestion

    left, right = suggestion, sorted_list.shape[0] - 1
    while right - left > 1:
        middle = (left + right) // 2
        if sorted_list[middle + 1] <= x:
            left = middle + 1
        elif x < sorted_list[middle - 1]:
            right = middle - 1
        elif sorted_list[middle] <= x:
            left = middle
        elif x < sorted_list[middle]:
            right = middle

    return left


# sorted_list 描述了一些“左闭右开区间”的右边界，返回下表标k iff a_{k} <= x < a_k+1
def search_linear(sorted_list, x, suggestion=0):
    while not sorted_list[suggestion] <= x < sorted_list[suggestion + 1]:
        suggestion = suggestion + 1
    return suggestion


# 用于避免重复在内存中载入数据集
dataset_buffered = dict()


class LD2011_2014(Dataset):
    DATASET_KEY = "LD2011_2014"
    """
    length: 序列长度，length为不包含y的长度
    """

    def __init__(self, length, csv_file=r"dataset/LD2011_2014.csv", transform=None, size=370):
        self.transform = transform
        self.length = length
        if self.DATASET_KEY not in dataset_buffered:
            dataset_buffered[self.DATASET_KEY] = np.array(
                pd.read_csv(csv_file, usecols=range(1, size + 1), dtype="float32",
                            delimiter=";", decimal=",").to_numpy(), order='F')
        self.dataset = dataset_buffered[self.DATASET_KEY]
        # self.counts = np.where(self.dataset, 0, 1).sum(axis=0) - length
        self.offsets = (self.dataset != 0).argmax(axis=0)
        self.counts = self.dataset.shape[0] - self.offsets - length
        self.counts[self.counts < 0] = 0  # 舍弃掉不足length + 1的数据
        self.total_counts = np.concatenate((np.asarray([0]), np.add.accumulate(self.counts)))

        if transform is not None:
            raise NotImplementedError("不支持transform")

    def __len__(self):
        return self.total_counts[-1]

    def __getitem__(self, index) -> T_co:
        col = search_sorted(self.total_counts, index)
        row_offset = index - self.total_counts[col] + self.offsets[col]

        return self.dataset[row_offset: row_offset + self.length, col], self.dataset[row_offset + self.length, col]

    def __iter__(self):
        def iterator(ld):
            suggestion = 0
            for index in range(self.total_counts[-1]):
                col = search_linear(self.total_counts, index, suggestion=suggestion)
                row_offset = index - self.total_counts[col] + self.offsets[col]

                suggestion = col
                yield ld.dataset[row_offset: row_offset + self.length, col], \
                      ld.dataset[row_offset + self.length, col]

        return iterator(self)


class LD2011_2014_summary(Dataset):
    DATASET_KEY = "LD2011_2014_summary"
    """
    length: 序列长度，length为不包含y的长度
    """

    def __init__(self, length, csv_file=r"../dataset/LD2011_2014.csv", transform=None, size=370):
        self.transform = transform
        self.length = length
        if self.DATASET_KEY not in dataset_buffered:
            dataset_buffered[self.DATASET_KEY] = np.array(pd.read_csv(csv_file, usecols=range(1, size + 1),
                                                                      dtype="float32", delimiter=";", decimal=",").
                                                          to_numpy(), order='F').sum(axis=1)
        self.dataset = dataset_buffered[self.DATASET_KEY]
        # self.counts = np.where(self.dataset, 0, 1).sum(axis=0) - length
        self.offsets = (self.dataset != 0).argmax(axis=0)
        self.counts = self.dataset.shape[0] - self.offsets - length

    def __len__(self):
        return self.counts

    def __getitem__(self, index) -> T_co:
        row_offset = index + self.offsets

        sample = self.dataset[row_offset: row_offset + self.length], self.dataset[row_offset + self.length]
        if self.transform:
            return self.transform(sample)
        else:
            return sample

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class LD2011_2014_summary_by_day(Dataset):
    T = 96

    DATASET_KEY = "LD2011_2014_summary_by_day"
    """
    length: 序列长度，length为不包含y的长度，T个单位为一个单位
    """

    def __init__(self, length, csv_file=r"../dataset/LD2011_2014.csv", transform=None, size=370):
        self.transform = transform
        self.length = length
        if self.DATASET_KEY not in dataset_buffered:
            dataset_buffered[self.DATASET_KEY] = np.array(pd.read_csv(csv_file, usecols=range(1, size + 1),
                                                                      dtype="float32", delimiter=";", decimal=",").
                                                          to_numpy(), order='F').sum(axis=1)
        self.dataset = dataset_buffered[self.DATASET_KEY]
        self.offsets = (self.dataset != 0).argmax(axis=0)
        self.counts = (self.dataset.shape[0] - self.offsets) // self.T - length

    def __len__(self):
        return self.counts

    def __getitem__(self, index) -> T_co:
        row_offset = index * self.T + self.offsets

        x, y = self.dataset[row_offset: row_offset + self.length * self.T], self.dataset[row_offset +
                                                                                         self.length * self.T]
        x = x.reshape(self.length, self.T)
        sample = x, y

        if self.transform:
            return self.transform(sample)
        else:
            return sample

    def __iter__(self):
        return (self[i] for i in range(len(self)))


RANDOM_SEED = 1023
BATCH_SIZE = 128


# X 成形于 [batch_size, length]
def construct_dataloader(raw_dataset, batch_size=BATCH_SIZE,
                         train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2):
    if train_ratio + validation_ratio + test_ratio != 1:
        raise ValueError("The sum of train_ratio, validation_ratio and test_ratio doesn't equal to One.")

    train_dataset, val_dataset, test_dataset = random_split(
        dataset=raw_dataset,
        lengths=[round(train_ratio * len(raw_dataset)),
                 round(validation_ratio * len(raw_dataset)),
                 len(raw_dataset) - round(train_ratio * len(raw_dataset)) -
                 round(validation_ratio * len(raw_dataset))],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    return DataLoader(train_dataset, batch_size=batch_size), DataLoader(val_dataset, batch_size=batch_size), \
           DataLoader(test_dataset, batch_size=batch_size)
