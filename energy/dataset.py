from math import ceil

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

import pandas as pd
import numpy as np

from itertools import accumulate


# 返回sorted_list中的区间序号
# sorted_list 描述了一些“左闭右开区间”的右边界，返回下表标k iff a_{k} <= x < a_k+1
def search_sorted(sorted_list, x, suggestion=0):
    # for i, count in enumerate(sorted_list):
    #     if x < count:
    #         return i

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
    """
    length: 序列长度，length为不包含y的长度
    """

    def __init__(self, length, csv_file=r"dataset/LD2011_2014.csv", transform=None, size=370):
        self.transform = transform
        self.length = length
        if "LD2011_2014" not in dataset_buffered:
            dataset_buffered["LD2011_2014"] = np.array(pd.read_csv(csv_file, usecols=range(1, size + 1), dtype="float32",
                                                              delimiter=";", decimal=",").to_numpy(), order='F')
        self.dataset = dataset_buffered["LD2011_2014"]
        # self.counts = np.where(self.dataset, 0, 1).sum(axis=0) - length
        self.offsets = (self.dataset != 0).argmax(axis=0)
        self.counts = self.dataset.shape[0] - self.offsets - length
        self.counts[self.counts < 0] = 0  # 舍弃掉不足length + 1的数据
        self.total_counts = np.concatenate((np.asarray([0]), np.add.accumulate(self.counts)))

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

    """
    length: 序列长度，length为不包含y的长度
    """
    def __init__(self, length, csv_file=r"dataset/LD2011_2014.csv", transform=None, size=370):
        self.transform = transform
        self.length = length
        if "LD2011_2014_summary" not in dataset_buffered:
            dataset_buffered["LD2011_2014_summary"] = np.array(pd.read_csv(csv_file, usecols=range(1, size + 1),
                                                                           dtype="float32", delimiter=";", decimal=",").
                                                               to_numpy(), order='F').sum(axis=1)
        self.dataset = dataset_buffered["LD2011_2014_summary"]
        # self.counts = np.where(self.dataset, 0, 1).sum(axis=0) - length
        self.offsets = (self.dataset != 0).argmax(axis=0)
        self.counts = self.dataset.shape[0] - self.offsets - length

    def __len__(self):
        return self.counts

    def __getitem__(self, index) -> T_co:
        row_offset = index + self.offsets

        return self.dataset[row_offset: row_offset + self.length], self.dataset[row_offset + self.length]

    def __iter__(self):
        return (self[i] for i in range(len(self)))