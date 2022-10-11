# -*- coding: UTF-8 -*- #
"""
@filename:london_merge.py
@author:201300086
@time:2022-10-10
"""
import pandas as pd
import numpy as np
# df=pd.read_csv('dataset/london_merge_all.csv', header=0, decimal=",",
#                na_filter=False,)
# df_date=df[df.columns[0]]
# df_date=df_date.drop(df_date.index[:1920],axis=0)
# print(df_date)


# 输出
# outputpath='dataset/london_date.csv'
# df_date.to_csv(outputpath,sep=',',index=False)
# df_data=df[df.columns[1:]]
# df_data=df_data.drop(df_data.index[:1920],axis=0)
# print(df_data)
# # 输出
# outputpath='dataset/london_data.csv'
# df_data.to_csv(outputpath,sep=',',index=False)


#数据集非0部分分布
data=np.load('dataset/london_data.npy')
data=data.astype(bool).astype(int)
data_distribute=np.sum(data,axis=1)
print(data_distribute.shape)
print(data_distribute)#d[5280]=5011,d[5279]=3272
# i=5000
# while(data_distribute[i]<4000):
#     i+=1
# print(i)
# print(data_distribute[i])
# print(data_distribute[i-1])
import matplotlib
import matplotlib.pyplot as plt
dataset_axis = np.arange(data_distribute.shape[0])
#dataset绘图
plt.plot(dataset_axis, data_distribute, c='red', label='num_of_users')
plt.legend()
plt.show()