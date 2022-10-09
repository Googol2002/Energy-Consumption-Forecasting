# -*- coding: UTF-8 -*- #
"""
@filename:TestLondon.py
@author:201300086
@time:2022-10-06
"""
import random
import pytest
import numpy as np
import matplotlib.pyplot as plt

from energy.dataset.london_clean import London_11_14

dataset = London_11_14(train_l=5, test_l=1, size=30)
"""
    train_l：训练集天数
    test_l：测试集天数
    总天数约800+
    size：随机可重复抽取样本的个数，可修改范围：10~5498
"""
print(dataset.data_all)
#print(dataset.dataset)
print(dataset.days)
# print(dataset.__len__())
# print(dataset.__getitem__(dataset.__len__() - 1))
print(len(dataset))
print(dataset[len(dataset)-1])
"""
    __getitem__()传参范围range(dataset.__len__())
    越界会assert
"""
