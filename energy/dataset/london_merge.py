# -*- coding: UTF-8 -*- #
"""
@filename:london_merge.py
@author:201300086
@time:2022-10-10
"""
import pandas as pd
df=pd.read_csv('dataset/london_merge_all.csv', header=0, decimal=",",
               na_filter=False,)
# df_date=df[df.columns[0]]
# df_date=df_date.drop(df_date.index[:1920],axis=0)
# print(df_date)
# # 输出
# outputpath='dataset/london_date.csv'
# df_date.to_csv(outputpath,sep=',',index=False)
df_data=df[df.columns[1:]]
df_data=df_data.drop(df_data.index[:1920],axis=0)
print(df_data)
# 输出
outputpath='dataset/london_data.csv'
df_data.to_csv(outputpath,sep=',',index=False)