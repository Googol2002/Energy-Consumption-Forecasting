# -*- coding: UTF-8 -*- #
"""
@filename:TestLondon.py
@author:201300086
@time:2022-10-06
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import time
import pandas as pd

from energy.dataset.london_clean import London_11_14,London_11_14_random_select

# dataset = London_11_14(train_l=5, test_l=1, size=1000)

# test:
# df=London_11_14(train_l=5, test_l=1, size=200).before_sum
# print(df)

#df_data=pd.read_csv('dataset/london_data.csv', header=0, decimal=",",na_filter=False,dtype='float32')
#data = np.genfromtxt('dataset/london_data.csv', dtype=float, delimiter=',', names=True)
t_begin = time.time()
data=np.load('dataset/london_data.npy')
df_data=pd.DataFrame(data)
df_date=pd.read_csv('dataset/london_date.csv', header=0, decimal=",",na_filter=False)
t_end1 = time.time()
print("time for load:", t_end1 - t_begin)


df=pd.merge(df_date,df_data,how='outer',right_index=True,left_index=True)
#df[df.columns[1:]] = df[df.columns[1:]].astype(float)
t_end2 = time.time()
print("time for concat:", t_end2 - t_end1)


t_begin = time.time()
dataset = London_11_14_random_select(df=df,train_l=5, test_l=1, size=3000)
t_end1 = time.time()
print("time for once size:", t_end1 - t_begin)

"""
    train_l：训练集天数
    test_l：测试集天数
    总天数约800+
    size：随机可重复抽取样本的个数，可修改范围：10~5498
"""
# print(dataset.data_all)
#print(dataset.dataset)
print(dataset.days)
print(len(dataset))
print(dataset[len(dataset)-1])
"""
    __getitem__()传参范围range(dataset.__len__())
    越界会assert
"""
START=0
END=dataset.days-1
dataset_axis = np.arange(dataset.dataset.shape[0])
plt.plot(dataset_axis[48*START:48*END], dataset.dataset[48*START:48*END], c='blue', label='X')

plt.legend()
    # plt.grid(True) # 显示网格线
    # plt.savefig("ARIMA.png")
plt.show()