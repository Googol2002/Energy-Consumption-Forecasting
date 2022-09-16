from math import ceil

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

import pandas as pd
import numpy as np

from itertools import accumulate


# 返回sorted_list中的区间序号
def search_sorted(sorted_list, x, suggestion=0):
    # for i, count in enumerate(sorted_list):
    #     if x < count:
    #         return i

    # 更新为Binary Search

    if not (0 <= x < sorted_list[-1]):
        raise IndexError()
    if sorted_list[suggestion] <= x < sorted_list[suggestion + 1]:
        return suggestion

    # sorted_list 描述了一些“左闭右开区间”的右边界，返回下表标k iff a_{k} <= x < a_k+1
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


# 用于避免重复在内存中载入数据集
dataset_buffered = dict()


class LD2011_2014(Dataset):
    """
    length: 序列长度，length为不包含y的长度
    """

    def __init__(self, length, csv_file=r"dataset/LD2011_2014.csv", transform=None):
        self.transform = transform
        self.length = length
        if csv_file not in dataset_buffered:
            dataset_buffered[csv_file] = np.array(pd.read_csv(csv_file, usecols=range(1, 371), dtype="float32",
                                                              delimiter=";", decimal=",").to_numpy(), order='F')
        self.dataset = dataset_buffered[csv_file]
        self.counts = np.where(self.dataset, 0, 1).sum(axis=0) - length
        self.counts[self.counts < 0] = 0  # 舍弃掉不足length + 1的数据
        self.total_counts = np.concatenate((np.asarray([0]), np.add.accumulate(self.counts)))

    def __len__(self):
        return self.total_counts[-1]

    def __getitem__(self, index) -> T_co:
        col = search_sorted(self.total_counts, index)
        row_offset = index - self.total_counts[col]

        return self.dataset[row_offset: row_offset + self.length, col], self.dataset[row_offset + self.length, col]

    def __iter__(self):
        def iterator(ld):
            suggestion = 0
            for index in range(self.total_counts[-1]):
                col = search_sorted(self.total_counts, index, suggestion=suggestion)
                row_offset = index - self.total_counts[col]
                suggestion = col
                yield ld.dataset[row_offset: row_offset + self.length, col], \
                      ld.dataset[row_offset + self.length, col]

        return iterator(self)
        # return (self[i] for i in range(self.total_counts[-1]))
