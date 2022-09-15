from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

import pandas as pd
import numpy as np

from itertools import accumulate

# 返回sorted_list中的区间序号
def search_sorted(sorted_list, x):
    # TODO: 更新为Binary Search
    for i, count in enumerate(sorted_list):
        if x < count:
            return i
    raise KeyboardInterrupt("Total {}, but x={}".format(sorted_list[-1], x))


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
            dataset_buffered[csv_file] = np.array(pd.read_csv(csv_file, usecols=range(1, 20), dtype="float32",
                                                              delimiter=";", decimal=",").to_numpy(), order='F')
        self.dataset = dataset_buffered[csv_file]
        self.counts = np.where(self.dataset, 0, 1).sum(axis=0) - length
        self.counts[self.counts < 0] = 0    # 舍弃掉不足length + 1的数据
        self.total_counts = np.add.accumulate(self.counts)

        print(self.total_counts)

    def __len__(self):
        return self.total_counts[-1]

    def __getitem__(self, index) -> T_co:
        col = search_sorted(self.total_counts, index)
        row_offset = index if col == 0 else (index - self.total_counts[col - 1])

        return self.dataset[row_offset: row_offset + self.length, col], self.dataset[row_offset + self.length, col]

    def __iter__(self):
        return (self[i] for i in range(self.total_counts[-1]))