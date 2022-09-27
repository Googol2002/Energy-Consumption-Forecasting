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

import statsmodels.api as sm

sys.path.append('..')



# Load/split your data
from energy.dataset import LD2011_2014_summary
dataset = LD2011_2014_summary(0)
y = dataset.dataset
print(type(y),y.size,y.ndim)
train_T=10
test_T=15

#plt.plot(np.arange(yy.shape[0]),yy)


#差分使数据平稳
# yy_df = pd.DataFrame(yy).diff(3)#转化pd求差分，去除空值
# yy_df.dropna(inplace=True)#不能赋值，会变成NoneType
# yy=yy_df.values


#acf，pacf确认参数
yy=y[:100*96]
plot_acf(yy)
plot_pacf(yy)
plt.show()

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
train, test = y[:train_T*24*4], y[train_T*24*4:test_T*24*4]
# model = pm.auto_arima(train, d=1, start_p=1, start_q=1,seasonal=True, m=train_T,information_criterion='aic',stepwise=False)
model=sm.tsa.ARIMA(train,order=(2,0,0)).fit()

# 模型导出与载入
import joblib
# joblib.dump(model,'auto_arima.pkl')
# model = joblib.load(model,'auto_arima.pkl')

# make your forecasts
# predict N steps into the future
forecasts = model.predict(test.shape[0])[:test.shape[0]]

# Visualize the forecasts (blue=train, green=forecasts)
test_axis = np.arange(test.shape[0])
print(test.shape[0],forecasts.shape[0])
plt.plot(test_axis, test, c='blue')
plt.plot(test_axis, forecasts, c='green')
#plt.savefig("ARIMA.png")
plt.show()
