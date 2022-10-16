# -*- coding: UTF-8 -*- #
"""
@filename:TestLondon.py
@author:201300086
@time:2022-10-06
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import time
import pandas as pd

from energy.dataset.london_clean import London_11_14, London_11_14_random_select  # ,London_11_14_set_train
from energy.dataset.london_clean import London_11_14_set_test, London_11_14_set, createDataSet

# dataset = London_11_14(train_l=5, test_l=1, size=1000)

# test:
# df=London_11_14(train_l=5, test_l=1, size=200).before_sum
# print(df)

# df_data=pd.read_csv('dataset/london_data.csv', header=0, decimal=",",na_filter=False,dtype='float32')
# data = np.genfromtxt('dataset/london_data.csv', dtype=float, delimiter=',', names=True)
# t_begin = time.time()
# data=np.load('dataset/london_data.npy')
# df_data=pd.DataFrame(data)
# df_date=pd.read_csv('dataset/london_date.csv', header=0, decimal=",",na_filter=False)
# #t_end1 = time.time()
# #print("time for load:", t_end1 - t_begin)
# df=pd.merge(df_date,df_data,how='outer',right_index=True,left_index=True)
# df[df.columns[1:]] = df[df.columns[1:]].astype(float)
# t_end2 = time.time()
# print("time for concat:", t_end2 - t_end1)


dataset = London_11_14_random_select(train_l=7, test_l=7, size=3000)
# # print(len(dataset))
# # print(dataset[len(dataset)-1])
# expectations,variances=dataset.statistics(110,7)
# print("expectations:",expectations)
# print("variances:",variances)
# print(expectations.mean())
# print(variances.mean())
# print(dataset.dataset_all_mean)
# print(dataset.dataset_all_var)
# print(dataset.dataset_all_std)
# print(dataset.days)

"""
    __getitem__()传参范围range(dataset.__len__())
    越界会assert
"""
# print(type(dataset.dataset),dataset.dataset.shape)
# print(type(dataset))
"""
    :param train_l：X天数
    :param label_l：y天数
    :param test_days：测试集组数（不参与数据增强），实际占天数label_l*test_days
    :param test_continuous：每组测试集连续天数，默认1
    :param times: 重复抽样次数，10次大致对应3000个元组(x, y, x_1, y_1)
    :param size: 随机抽取的用户数量，上限5068
    :param ev_key: 期望和方差的统计长度，=1代表一天48列，=7代表一周48*7列
    测试集个数=label_l*test_days*test_continuous
"""
set1, set2, expectations, variances = createDataSet(train_l=10, label_l=7, test_days=3, test_continuous=5, size=3000,
                                                    times=1, ev_key=1)
print("expectations_mean:", expectations.mean())
print("variances_mean:", variances.mean())
# print(set2[0])
print("train_lens:", len(set1), "test_lens:", len(set2))
#print(set1[1])

# data_set=London_11_14_set(train_l=5, label_l=1, size=3000,times=10)#time for set: 11.827157974243164
# print(data_set[len(data_set)-1])#getitem访问
# print(len(data_set))
# expectations,variances=data_set.statistics()
# print("expectations:",expectations)
# print("variances:",variances)


# print(data_set.lst[1])#访问元组示例:(x, y, x_1, y_1)
# print(data_set.arr.shape)#(4820, 4)
# print(data_set.arr[1])#<class 'numpy.ndarray'>[x, y, x_1, y_1]
# print(data_set.counts)#元组个数:4820=10*(488-5-1)


# print(data_set.set.dataset.shape)#已对纯数据取并集，所以不再是48的倍数
# expectations = np.mean(data_set.set.dataset.reshape(-1, 48), axis=0)
# variances = np.var(data_set.set.dataset.reshape(-1, 48), axis=0)
# print(expectations)
# print(variances)

##直接用日或周均值来预测的效果
ev_key=7
accuracy = []
for j in range(5):
    dataset = London_11_14_random_select(train_l=7, test_l=7, size=3000)
    accuracy_t = 0
    for i in range(len(dataset)):
        x, y, x1, y1 = dataset[i]
        e,_=dataset.statistics(ev_key)#=1代表一天，=7代表一周
        e=e.reshape(-1,48)
        accuracy_t += np.mean(1 - np.abs((e - y) / y))
    print("[{}/{}]".format(j+1, 10))

    accuracy_t /= len(dataset)
    accuracy_t *= 100
    accuracy.append(accuracy_t)

print(f"Accuracy: {accuracy} \nMean:{np.mean(accuracy)} using {ev_key} days of expectations")

# START=0
# END=dataset.days-1
# dataset_axis = np.arange(dataset.dataset.shape[0])
# #dataset绘图
# plt.plot(dataset_axis[48*START:48*END], dataset.dataset[48*START:48*END], c='blue', label='X')
# # #均值绘图
# # plt.plot(dataset_axis[48*START:48*END:48], dataset.dataset_mean[START:END], c='green', label='X_mean')
# # #非0分布绘图
# # data=np.load('dataset/london_data.npy')
# # data=data.astype(bool).astype(int)
# # data_distribute=np.sum(data,axis=1)
# # plt.plot(dataset_axis, data_distribute, c='red', label='counts')
# plt.legend()
#     # plt.grid(True) # 显示网格线
#     # plt.savefig("ARIMA.png")
# plt.show()
