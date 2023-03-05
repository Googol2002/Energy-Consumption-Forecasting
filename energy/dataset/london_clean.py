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
import time

import copy
import torch

# import tensorflow as tf
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SIZE = 10
TIMES = 10
Train_length = 10
Test_length = 7


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
        success_read_file = []

        for f in select_file_list:
            file_name = "cleaned_household_MAC0" + str(f) + ".csv"
            file_name = os.path.join(LOG_DIRECTORY, file_name).replace('\\', '/')
            if os.path.exists(file_name):
                df_temp = pd.read_csv(file_name, header=0, usecols=[4, 3], decimal=",")
                df_temp[df_temp.columns[-1]] = df_temp[df_temp.columns[-1]].astype(float)
                if df_temp.shape[0] > 20000:
                    success_read_file.append(df_temp)  # 读取每个表格

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
        self.before_sum = df_merge

        # 求和
        t_begin = time.time()
        df_merge['sum'] = df_merge.sum(axis=1, numeric_only=True)
        df_merge = df_merge.drop(df_merge.columns[1:-1], axis=1)
        t_end = time.time()

        # while (df_merge.shape[1] >= 3):
        #     df_merge[df_merge.columns[1]] = df_merge[df_merge.columns[1]] + \
        #                                     df_merge[df_merge.columns[-1]]
        #     df_merge = df_merge.drop(df_merge.columns[-1], axis=1)

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
        self.dataset = np.array(self.data_only.values.tolist()).astype("float32")
        self.data_week = np.array(self.data_all[self.data_all.columns[2]].values.tolist())
        self.data_month = np.array(self.data_all[self.data_all.columns[3]].values.tolist())
        self.days = int(self.data_only.shape[0] / 48)
        self.counts = self.days - test_l - train_l + 1  # (X,y)总行数

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
                        self.data_month[row_offset: row_offset + self.train_l * 48:48], axis=1)

        y_1 = np.append(
            self.data_week[row_offset + self.train_l * 48: row_offset + (self.train_l + self.test_l) * 48:48],
            self.data_month[row_offset + self.train_l * 48: row_offset + (self.train_l + self.test_l) * 48:48], axis=1)

        x_1 = x_1.reshape(self.train_l, 19)
        y_1 = y_1.reshape(self.test_l, 19)
        return x, y, x_1, y_1


class London_11_14_random_select(Dataset):
    """
    :param df:一次读取，多次抽取，在类外读取后传入类
    :param size: 随机抽取的用户数量，上限5068
    """

    def __init__(self, train_l=Train_length, test_l=Test_length, size=SIZE):

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
            week = list(df[df.columns[0]])
            Week = []
            for i in week:
                a = int(i[0:4])
                b = int(i[5:7])
                c = int(i[8:10])
                week_num = datetime.date(a, b, c).isoweekday()
                Week.append(get_one_hot(week_num, 7))
            df['Week'] = Week

        def add_one_hot_month(df):
            month = list(df[df.columns[0]])
            Month = []
            for i in month:
                j = int(i[5:7])
                Month.append(get_one_hot(j, 12))
            df['Month'] = Month

        self.train_l = train_l
        self.test_l = test_l
        self.size = size
        df_data = pd.DataFrame(np.load('dataset/london_data.npy'))
        df_date = pd.read_csv('dataset/london_date.csv', header=0, decimal=",", na_filter=False)
        self.df = pd.merge(df_date, df_data, how='outer', right_index=True, left_index=True)
        self.df = self.df.drop(self.df.index[:110 * 48], axis=0)  # 前110天只有3000户有值，后面都是5000户，删！
        self.init_columns = self.df.shape[1] - 1
        self.delete_columns = self.init_columns - self.size

        # 随机选取
        assert (self.size < self.init_columns)
        delete_list = random.sample(list(range(1, self.init_columns)), self.delete_columns)
        self.df = self.df.drop(self.df.columns[delete_list], axis=1)
        self.df['summ'] = self.df.sum(axis=1, numeric_only=True)
        self.df = self.df.drop(self.df.columns[1:-1], axis=1)
        add_one_hot_week(self.df)  # 添加周独热编码
        add_one_hot_month(self.df)  # 添加月独热编码
        self.data_only = self.df[self.df.columns[1]]
        self.dataset = (np.array(self.data_only.values.tolist()))  # 只保留用电量数据
        # self.dataset_mean=np.mean(self.dataset.reshape(-1,48), axis=1)
        # self.dataset_all_mean=np.mean(self.dataset_mean)
        # self.dataset_all_var=np.var(self.dataset)
        # self.dataset_all_std = np.std(self.dataset)
        self.data_week = (np.array(self.df[self.df.columns[2]].values.tolist()))
        self.data_month = (np.array(self.df[self.df.columns[3]].values.tolist()))
        self.days = int(self.data_only.shape[0] / 48)
        self.counts = self.days - test_l - train_l + 1  # (X,y)总行数

        # 输出
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
                        self.data_month[row_offset: row_offset + self.train_l * 48:48], axis=1)

        y_1 = np.append(
            self.data_week[row_offset + self.train_l * 48: row_offset + (self.train_l + self.test_l) * 48:48],
            self.data_month[row_offset + self.train_l * 48: row_offset + (self.train_l + self.test_l) * 48:48], axis=1)

        x_1 = x_1.reshape(self.train_l, 19)
        y_1 = y_1.reshape(self.test_l, 19)
        return x, y, x_1, y_1

    def __add__(self, other):
        return pd.concat(self, other)

    def statistics(self, lens):
        """
        :param lens: 期望方差的长度（一天48，一周7*48）
        """
        # assert (skipdays < self.counts)
        # a = np.delete(self.dataset, np.s_[:skipdays*48], axis=0)
        offset = self.dataset.shape[0] % (48 * lens)
        # print('unused days for e and v:',int(offset/48),'days')
        if offset != 0:
            a = np.delete(self.dataset, np.s_[-offset:], axis=0)
            c = a.reshape(-1, 48 * lens)
        else:
            c = self.dataset.reshape(-1, 48 * lens)
        # expectations =tf.reduce_mean(,axis=0)
        expectations = np.mean(c, axis=0)
        # variances = tf.reduce_mean(tf.cast(a, tf.float32), axis=0)
        variances = np.var(c, axis=0)
        return expectations, variances


class London_11_14_set(London_11_14_random_select):
    """
    :param train_l：X天数
    :param label_l：y天数
    :param test_days：测试集组数（不参与数据增强），实际占天数label_l*test_days
    :param test_continuous：每组测试集连续天数，默认1
    :param times: 训练集重复抽样次数，10次大致对应3000个元组(x, y, x_1, y_1)
    :param size: 随机抽取的用户数量，上限5068
    :param test_list: 需要在训练集去除的样本
    :param ev_key: 期望和方差的统计意义，=1代表一天48列，=7代表一周48*7列
    """

    def __init__(self, train_l=Train_length, label_l=Test_length, test_days=10,
                 test_continuous=1, size=SIZE, times=TIMES, test_list=None, data_list=None, ev_key=1):
        self.train_l = train_l
        self.label_l = label_l
        self.test_continuous = test_continuous
        self.test_days = test_days * (label_l + self.test_continuous - 1)  # test实际占据天数
        self.size = size
        self.times = times
        self.test_list = test_list if test_list else []  # 解决list=[]传递异常
        self.data_list = data_list if data_list else []
        self.expectations, self.variances = 0.0, 0.0

        def merge_e(x, m, y, n):
            return (x * m + y * n) / (m + n)

        def merge_v(x, m, y, n, x_mean, y_mean):
            a = m * x + n * y
            b = (m * n * (x_mean - y_mean) * (x_mean - y_mean)) / (m + n)
            return (a + b) / (m + n)

        self.lst = []

        # 判断当前index加入train后是否造成test泄露
        def conflict_with_test(index, size, lst):
            for i in range(size):
                if index + i in lst:
                    return True
            return False

        for i in range(self.times):
            other = self.data_list[i]
            e, v = other.statistics(ev_key)
            self.variances = merge_v(self.variances, len(self.lst), v, other.counts, self.expectations, e)
            self.expectations = merge_e(self.expectations, len(self.lst), e, other.counts)

            for j in range(len(other)):
                if (conflict_with_test(j, self.label_l + self.test_continuous - 1, self.test_list) == False):
                    # if j not in self.test_list:
                    self.lst.append(other[j])

        self.arr = np.array(self.lst, dtype=object)
        self.counts = len(self.lst)

    def __len__(self):
        return self.counts

    def __getitem__(self, index):
        assert (index < self.__len__())
        x, y, x_1, y_1 = self.lst[index]
        return x, y, x_1, y_1

    def statistics(self):
        return self.expectations, self.variances


class London_11_14_set_test(Dataset):
    """
    :param flod:第k折交叉验证得到的数据集，默认为0，总范围range(0,k_flod)
    :param train_l：X天数
    :param label_l：y天数
    :param test_days：测试集组数（不参与数据增强），实际占天数label_l*test_days
    :param test_continuous：每组测试集连续天数，默认1
    :param times: 测试集重复抽样次数
    :param size: 随机抽取的用户数量，上限5068
    """

    def __init__(self, flod=0, k_flod=20, train_l=Train_length, label_l=Test_length, test_days=10,
                 test_continuous=1, size=SIZE, times=TIMES, k_flod_test_list=None):
        # super().__init__(train_l=Train_length, label_l=Test_length, test_days=70, size=SIZE, times=TIMES)
        self.flod = flod
        self.train_l = train_l
        self.label_l = label_l
        self.test_continuous = test_continuous
        self.test_groups = test_days
        self.test_days = test_days * (label_l + self.test_continuous - 1)  # test实际占据天数
        self.size = size
        self.times = times
        self.days = 378 - self.train_l - self.label_l
        self.train_days = self.days - self.test_days
        self.k_flod_test_list = k_flod_test_list if k_flod_test_list else []  # 已随机划分好的基准k折测试集
        self.test_list = (np.array(self.k_flod_test_list) + self.flod * self.test_continuous).tolist()
        print("test_list:", self.test_list)
        self.data_test = []
        other = London_11_14_random_select(train_l=self.train_l, test_l=self.label_l, size=self.size)
        # 取出测试集
        self.data_lst = []  # 存储times次数据集
        for i in range(self.times):
            other = London_11_14_random_select(train_l=self.train_l, test_l=self.label_l, size=self.size)
            self.data_lst.append(other)
            for k in range(len(self.test_list)):
                for m in range(self.test_continuous):
                    self.data_test.append(other[self.test_list[k] - m])
            self.arr = np.array(self.data_test, dtype=object)
            self.counts = len(self.data_test)

    def __len__(self):
        return len(self.data_test)

    def __getitem__(self, index):
        assert (index == 0 or index < self.__len__())
        x, y, x_1, y_1 = self.data_test[index]
        return x, y, x_1, y_1

    def get_test_list(self):  # 用于传递测试集index，防止泄露在train中
        return self.test_list

    def get_data_list(self):  # 用于传递times次抽取的数据集
        return self.data_lst


# 尝试一个记录时间装饰器，事实上关键字参数不好传递？
def record_time(func):
    def wrapper(*args, **kwargs):  # 包装带参函数
        start_time = time.perf_counter()
        a = func(*args, **kwargs)  # 包装带参函数
        end_time = time.perf_counter()
        print('time=', end_time - start_time)
        return a  # 有返回值的函数必须给个返回

    return wrapper


@record_time
def createDataSet(k_flod=20, train_l=Train_length, label_l=Test_length, test_days=10,
                  test_continuous=1, size=SIZE, times=TIMES, ev_key=1):
    """
        :param k_flod:k折交叉验证
        :param train_l：X天数
        :param label_l：y天数
        :param test_days：测试集组数（不参与数据增强），实际占天数label_l*test_days
        :param test_continuous：每组测试集连续天数，默认1
        :param times: 训练集和测试集的重复抽样次数
        :param size: 随机抽取的用户数量，上限5068
        :param ev_key: 期望和方差的统计意义，=1代表一天48列，=7代表一周48*7列
    """
    k_flod_test_list = (np.array(sorted(
        random.sample(
            list(range(test_continuous, 378 - train_l - label_l - k_flod * test_continuous + 1, test_continuous)),
            test_days)
    ))).tolist()  # 已随机划分好的基准k折测试集
    print("k_flod_test_list:", k_flod_test_list)
    set2_flod = []
    set1_flod = []
    e_flod = []
    v_flod = []
    for flod in range(k_flod):
        set2 = London_11_14_set_test(flod=flod, k_flod=k_flod, train_l=train_l, label_l=label_l, test_days=test_days,
                                     test_continuous=test_continuous, size=size, times=times,
                                     k_flod_test_list=k_flod_test_list, )
        set1 = London_11_14_set(train_l=train_l, label_l=label_l, test_days=test_days, test_continuous=test_continuous,
                                size=size, times=times, test_list=set2.get_test_list(), data_list=set2.get_data_list(),
                                ev_key=ev_key)
        # print("train_l=", train_l, "label_l=", label_l, "test_days=", test_days, "test_continuous=", test_continuous,
        #       'size=', size, 'times=', times, "ev_key=", ev_key)
        e, v = set1.statistics()
        set2_flod.append(set2)
        set1_flod.append(set1)
        e_flod.append(e)
        v_flod.append(v)
    return set1_flod, set2_flod, e_flod, v_flod


def createDataSetSingleFold(**kwargs):
    set1_flod, set2_flod, e_flod, v_flod = createDataSet(k_flod=1, **kwargs)
    return set1_flod[0], set2_flod[0], e_flod[0], v_flod[0]



if __name__ == '__main__':
    set1, set2, e, v = createDataSet(k_flod=10, train_l=Train_length, label_l=Test_length, test_days=10,
                                     test_continuous=3, size=SIZE, times=TIMES, ev_key=1)
