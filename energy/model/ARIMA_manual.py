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
train_T=10
test_T=16

train, test = y[:train_T*24*4], y[train_T*24*4:test_T*24*4]
#model = pm.auto_arima(train,test='adf',trace=True,seasonal=True,m=train_T)#
model = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 2, 1, train_T)).fit(disp=-1)
# model = pm.auto_arima(train, start_p=0, start_q=0,
#                       seasonal=True, m=train_T,
#                       information_criterion='aic',
#                       start_P=0,start_Q=0,D=None,trace=True,
#                       error_action='ignore',test='adf',
#                       d=None,
#                       suppress_warnings=True,stepwise=True)
#model=sm.tsa.ARIMA(train,order=(2,0,0)).fit()
#model=statsmodels.tsa.ar_model.AutoReg(train,2).fit()
#print(model.summary())

# 模型导出与载入
import joblib
# joblib.dump(model,'auto_arima.pkl')
# model = joblib.load(model,'auto_arima.pkl')

# make your forecasts
# predict N steps into the future
test_axis = np.arange(test.shape[0])

#forecasts = model.predict(test.shape[0])
forecasts = model.forecast(test.shape[0])

#mse误差
rmse = np.sqrt(mse(test, forecasts))
print('RMSE: %.4f' % rmse)

# Visualize the forecasts (blue=test, green=forecasts)
#print(test.shape[0])
print(test.shape[0],forecasts.shape[0])
plt.plot(test_axis, test, c='blue',label='test')
plt.plot(test_axis, forecasts, c='green',label='forecasts')
plt.legend()
#plt.grid(True) # 显示网格线
#plt.savefig("ARIMA.png")
plt.show()
