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

SIZE = 100  # 随机读取规模

select_list = [random.randint(2, 5500) for i in range(SIZE)]  # 随机读取SIZE个文件
select_file_list = [f"{i:0>5d}" for i in select_list]
# print('select_list:',select_list)


success_read_file = []
for f in select_file_list:
    file_name = "../london_clean/cleaned_household_MAC0" + str(f) + ".csv"
    print(file_name)
    if os.path.exists(file_name):
        success_read_file.append(pd.read_csv(file_name, header=0, usecols=[4, 3], decimal=","))  # 读取每个表格

# 初始化求和表
df_merge = pd.read_csv(r"../london_clean/cleaned_household_MAC000002.csv",
                       header=0, usecols=[4, 3], decimal=",")

# 合并
values = ['_' + str(i) for i in range(6000)]  # 用于merge中重复列的重命名
for i in range(len(success_read_file)):
    df_merge = pd.merge(df_merge, success_read_file[i], how='outer', on='DateTime', sort=True,
                        suffixes=('', values[i])).replace(
        np.nan, 0)
df_merge[df_merge.columns[1]] = 0
# print(df_merge)

# 求和
while (df_merge.shape[1] >= 3):
    df_merge[df_merge.columns[1]] = df_merge[df_merge.columns[1]].map(float) + \
                                    df_merge[df_merge.columns[-1]].map(float)
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
        print()
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

# 输出
print(df_merge)
# outputpath='../dataset/temp.csv'
# df_merge.to_csv(outputpath,sep=',',index=False)
