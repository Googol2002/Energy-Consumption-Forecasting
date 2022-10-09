# -*- coding: UTF-8 -*- #
"""
@filename:london_clean.py
@author:201300086
@time:2022-10-06
"""
import pandas as pd
import numpy as np
import glob, os
import random
import datetime
from torch.utils.data import Dataset, DataLoader

"""
最后的输出格式：X, y,X_1,y_1
X:L * W, y:L' * W,X_1:L * 19, y_1:L' * 19
L 是序列长，L'是答案长,W是24h内的数据点(48)
"""

SIZE = 10
Train_length = 10
Test_length = 3


class London_11_14(Dataset):
    """
    train_l：训练集天数
    test_l：测试集天数
    总天数约800+
    size：随机可重复抽取样本的个数，可修改范围：10~5498
    """

    def __init__(self, train_l=Train_length, test_l=Test_length, size=SIZE):
        self.train_l = train_l
        self.test_l = test_l
        self.size = size

        select_list = [random.randint(2, 5500) for i in range(self.size)]  # 随机读取SIZE个文件
        select_file_list = [f"{i:0>5d}" for i in select_list]

        LOG_DIRECTORY = "dataset/london_clean"
        #LOG_DIRECTORY = "../../london_clean"
        success_read_file = []
        for f in select_file_list:
            file_name = "cleaned_household_MAC0" + str(f) + ".csv"
            file_name = os.path.join(LOG_DIRECTORY, file_name).replace('\\', '/')
            print(file_name)
            if os.path.exists(file_name):
                success_read_file.append(pd.read_csv(file_name, header=0, usecols=[4, 3], decimal=","))  # 读取每个表格

        # 初始化求和表
        file_name = "cleaned_household_MAC000002.csv"
        file_name = os.path.join(LOG_DIRECTORY, file_name).replace('\\', '/')
        df_merge = pd.read_csv(file_name, header=0, usecols=[4, 3], decimal=",")

        # 合并
        values = ['_' + str(i) for i in range(6000)]  # 用于merge中重复列的重命名
        for i in range(len(success_read_file)):
            df_merge = pd.merge(df_merge, success_read_file[i], how='outer', on='DateTime', sort=True,
                                suffixes=('', values[i])).replace(np.nan, 0)
        df_merge[df_merge.columns[1]] = 0
        print(df_merge)

        # 求和
        while (df_merge.shape[1] >= 3):
            df_merge[df_merge.columns[1]] = df_merge[df_merge.columns[1]].map(float) + \
                                            df_merge[df_merge.columns[-1]].map(float)
            df_merge[df_merge.columns[1]] = df_merge[df_merge.columns[1]].apply(lambda x: '%.4f' % x)
            df_merge = df_merge.drop(df_merge.columns[-1], axis=1)

        # 添加周&月独热编码
        def get_one_hot(index, size):
            '''
            获得一个one-hot的编码
            index:编码值
            size:编码长度
            '''
            one_hot = [0 for _ in range(1, size + 1)]
            one_hot[index - 1] = 1
            return one_hot

        def add_one_hot_week(df):
            week = list(df['DateTime'])
            Week = []
            for i in week:
                a = int(i[0:4])
                b = int(i[5:7])
                c = int(i[8:10])
                week_num = datetime.date(a, b, c).isoweekday()
                Week.append(get_one_hot(week_num, 7))
            df['Week'] = Week

        def add_one_hot_month(df):
            month = list(df['DateTime'])
            Month = []
            for i in month:
                j = int(i[5:7])
                Month.append(get_one_hot(j, 12))
            df['Month'] = Month

        add_one_hot_week(df_merge)  # 添加周独热编码
        add_one_hot_month(df_merge)  # 添加月独热编码

        self.data_all = df_merge
        self.data_only = df_merge[df_merge.columns[1]]  # 只保留用电量数据
        self.dataset = np.array(self.data_only.values.tolist())
        self.data_week = np.array(self.data_all[self.data_all.columns[2]].values.tolist())
        self.data_month = np.array(self.data_all[self.data_all.columns[3]].values.tolist())
        self.days = int(self.data_only.shape[0] / 48)
        self.counts = self.days - test_l - train_l + 1  # (X,y)总行数

        # 输出
        # print(df_merge)
        # outputpath='../../dataset/example.csv'
        # df_merge.to_csv(outputpath,sep=',',index=False)

    def __len__(self):
        return self.counts

    def __getitem__(self, index):
        assert (index < self.__len__())
        row_offset = index * 48
        x, y = self.dataset[row_offset: row_offset + self.train_l * 48], \
               self.dataset[row_offset + self.train_l * 48: row_offset + (self.train_l + self.test_l) * 48]
        x = x.reshape(self.train_l, 48)
        y = y.reshape(self.test_l, 48)
        x_1 = np.append(self.data_week[row_offset: row_offset + self.train_l * 48:48],
                        self.data_month[row_offset: row_offset + self.train_l * 48:48],axis=1)

        y_1 =np.append(self.data_week[row_offset + self.train_l * 48: row_offset + (self.train_l + self.test_l) * 48:48],
                        self.data_month[row_offset + self.train_l * 48: row_offset + (self.train_l + self.test_l) * 48:48],axis=1)

        x_1 = x_1.reshape(self.train_l, 19)
        y_1 = y_1.reshape(self.test_l, 19)
        return x, y,x_1,y_1

