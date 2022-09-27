# -*- coding: UTF-8 -*- #
"""
@filename:ARIMA_manual.py
@author:201300086
@time:2022-09-27
"""
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sys

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error as mse
sys.path.append('..')



# Load/split your data
from energy.dataset import LD2011_2014_summary
dataset = LD2011_2014_summary(0)
y = dataset.dataset
print(type(y),y.size,y.ndim)

#查看部分数据走势
# yy=y[:200*96]
# plt.plot(np.arange(yy.shape[0]),yy)
# plt.show()

#差分使数据平稳
# yy_df = pd.DataFrame(yy).diff(3)#转化pd求差分，去除空值
# yy_df.dropna(inplace=True)#不能赋值，会变成NoneType
# yy=yy_df.values


#acf，pacf确认参数
# yy=y[:100*96]
# plot_acf(yy)
# plot_pacf(yy)
# plt.show()

#adfuller单位根检验平稳性
# from statsmodels.tsa.stattools import adfuller
# def adf_val(ts):
#     '''
#     ts: 时间序列数据，Series类型
#     ts_title: 时间序列图的标题名称，字符串
#     '''
#     # 稳定性（ADF）检验
#     adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(ts)
#
#     name = ['adf', 'pvalue', 'usedlag',
#             'nobs', 'critical_values', 'icbest']
#     values = [adf, pvalue, usedlag, nobs,
#               critical_values, icbest]
#     print(list(zip(name, values)))
#
#     return adf, pvalue, critical_values,
    # 返回adf值、adf的p值、三种状态的检验值

# yy = pd.Series(yy.tolist())#ndarry转series
# print(adfuller(yy))

#模型训练
train_T=20
test_T=25

import time
train, test = y[:train_T*24*4], y[train_T*24*4:test_T*24*4]

#model = pm.auto_arima(train,test='adf',trace=True,seasonal=True,m=train_T,D=2,max_p=4,max_q=4)#（201，220）（601，211）
model = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 2, 1, train_T)).fit(disp=-1)#最佳参数（201，121）
# model = pm.auto_arima(train, start_p=0, start_q=0,
#                       seasonal=True, m=train_T,
#                       information_criterion='aic',d=None,
#                       start_P=0,start_Q=0,D=None,trace=True,
#                       error_action='ignore',test='adf',
#                       suppress_warnings=True,stepwise=True)
print(model.summary())#自动寻参结束后显示模型详细信息

time_begin = time.time()

#auto用predict
#forecasts = model.predict(test.shape[0])
#手动用forecast
forecasts = model.forecast(test.shape[0])
time_end = time.time()
print('predict time:', time_end - time_begin)


#mse误差
rmse = np.sqrt(mse(test, forecasts))
print('RMSE: %.4f' % rmse)

#bias偏置
bias=0
for i in range(96):
    bias+=abs(forecasts[i]-test[i])/test[i]
bias/=test.shape[0]
bias*=100
print('bias: %.4f' % bias,'%')


#可视化
print(test.shape[0],forecasts.shape[0])
test_axis = np.arange(test.shape[0])
plt.plot(test_axis, test, c='blue',label='test')
plt.plot(test_axis, forecasts, c='green',label='forecasts')
plt.legend()
#plt.grid(True) # 显示网格线
#plt.savefig("ARIMA.png")
plt.show()

