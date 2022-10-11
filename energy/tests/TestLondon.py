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

from energy.dataset.london_clean import London_11_14,London_11_14_random_select,London_11_14_set

# dataset = London_11_14(train_l=5, test_l=1, size=1000)

# test:
# df=London_11_14(train_l=5, test_l=1, size=200).before_sum
# print(df)

#df_data=pd.read_csv('dataset/london_data.csv', header=0, decimal=",",na_filter=False,dtype='float32')
#data = np.genfromtxt('dataset/london_data.csv', dtype=float, delimiter=',', names=True)
#t_begin = time.time()
# data=np.load('dataset/london_data.npy')
# df_data=pd.DataFrame(data)
# df_date=pd.read_csv('dataset/london_date.csv', header=0, decimal=",",na_filter=False)
# #t_end1 = time.time()
# #print("time for load:", t_end1 - t_begin)
# df=pd.merge(df_date,df_data,how='outer',right_index=True,left_index=True)
#df[df.columns[1:]] = df[df.columns[1:]].astype(float)
#t_end2 = time.time()
#print("time for concat:", t_end2 - t_end1)


# dataset = London_11_14_random_select(train_l=5, test_l=1, size=3000)
# expectations,variances=dataset.statistics(110)
# print("expectations:",expectations)
# print("variances:",variances)
# print(expectations.mean())
# print(variances.mean())
# print(dataset.dataset_all_mean)
# print(dataset.dataset_all_var)
# print(dataset.dataset_all_std)
# print(dataset.days)
# print(len(dataset))
#print(dataset[len(dataset)-1])
"""
    __getitem__()传参范围range(dataset.__len__())
    越界会assert
"""
# print(type(dataset.dataset),dataset.dataset.shape)
#print(type(dataset))
"""
    :param train_l：训练集天数
    :param test_l：测试集天数
    :param times: 重复抽样次数，10次大致对应4000个元组(x, y, x_1, y_1)
    :param size: 随机抽取的用户数量，上限5068
    总元组数公式：times*(378-train_l-train_l)
    """
data_set=London_11_14_set(train_l=5, test_l=1, size=3000,times=10)#time for set: 11.827157974243164
print(data_set[len(data_set)-1])#getitem访问
print(len(data_set))
expectations,variances=data_set.statistics()
print("expectations:",expectations)
print("variances:",variances)
print(expectations.mean())
print(variances.mean())
# print(data_set.lst[1])#访问元组示例:(x, y, x_1, y_1)
# print(data_set.arr.shape)#(4820, 4)
# print(data_set.arr[1])#<class 'numpy.ndarray'>[x, y, x_1, y_1]
# print(data_set.counts)#元组个数:4820=10*(488-5-1)



#print(data_set.set.dataset.shape)#已对纯数据取并集，所以不再是48的倍数
# expectations = np.mean(data_set.set.dataset.reshape(-1, 48), axis=0)
# variances = np.var(data_set.set.dataset.reshape(-1, 48), axis=0)
# print(expectations)
# print(variances)


# START=0
# END=dataset.days-1
# dataset_axis = np.arange(dataset.dataset.shape[0])
# #dataset绘图
# plt.plot(dataset_axis[48*START:48*END], dataset.dataset[48*START:48*END], c='blue', label='X')
# #均值绘图
# plt.plot(dataset_axis[48*START:48*END:48], dataset.dataset_mean[START:END], c='green', label='X_mean')
# #非0分布绘图
# data=np.load('dataset/london_data.npy')
# data=data.astype(bool).astype(int)
# data_distribute=np.sum(data,axis=1)
# plt.plot(dataset_axis, data_distribute, c='red', label='counts')
# plt.legend()
#     # plt.grid(True) # 显示网格线
#     # plt.savefig("ARIMA.png")
# plt.show()