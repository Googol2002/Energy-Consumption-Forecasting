# -*- coding: UTF-8 -*- #
"""
@filename:ARIMA_best.py
@author:201300086
@time:2022-09-28
"""
# -*- coding: UTF-8 -*- #
"""
@filename:ARIMA_manual.py
@author:201300086
@time:2022-09-27
"""
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error as mse
sys.path.append('..')

# Load/split your data
from energy.dataset import LD2011_2014_summary
import time
dataset = LD2011_2014_summary(0)
y = dataset.dataset
print(type(y),y.size,y.ndim)

def forecast_bias(t:int,forecasts,test):
    # bias偏置
    bias = 0
    rmse=0
    for i in range(96):
        bias += abs(forecasts[i] - test[i]) / test[i]
        rmse+=(forecasts[i] - test[i])**2
    bias *=0.96
    rmse /=96
    rmse=np.sqrt(rmse)
    print('bias: %.4f' % bias,'%','  rmse: %d' % rmse, ' at train_T:',t,end="  ")
    return bias

#搜索最佳模型训练规模
min_bias=100
best_train_T=5
for train_T in range(22,45):
    test_T=train_T+3
    train, test = y[:train_T * 24 * 4], y[train_T * 24 * 4:test_T * 24 * 4]
    time_begin = time.time()
    model = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 2, 1, train_T),
                    ).fit(disp=-1)  # 最佳参数（201，121）
    time_end = time.time()
    forecasts = model.forecast(test.shape[0])
    temp_bias=forecast_bias(train_T,forecasts,test)
    print('train time: %.4f' % (time_end - time_begin))
    if temp_bias<min_bias:
        min_bias=temp_bias
        best_train_T=train_T
print('######## HIT best bias !!!:',min_bias,' at train_T:',best_train_T)

"""
bias: 40.2804 %   rmse: 62912  at train_T: 5  train time: 1.0466
bias: 13.9019 %   rmse: 30195  at train_T: 6  train time: 1.2933
bias: 17.2251 %   rmse: 27892  at train_T: 7  train time: 2.2698
bias: 175.3401 %   rmse: 278532  at train_T: 8  train time: 2.6321
bias: 28.6030 %   rmse: 43138  at train_T: 9  train time: 3.5808
bias: 19.3763 %   rmse: 31576  at train_T: 10  train time: 5.1662
bias: 14.1695 %   rmse: 26317  at train_T: 11  train time: 7.8003
bias: 32.2086 %   rmse: 44587  at train_T: 12  train time: 10.5368
bias: 126.1784 %   rmse: 208596  at train_T: 13  train time: 11.3913
bias: 14.3154 %   rmse: 19231  at train_T: 14  train time: 15.2625

bias: 17.5408 %   rmse: 29856  at train_T: 16  train time: 19.2998
bias: 18.5791 %   rmse: 32813  at train_T: 17  train time: 27.2111
bias: 16.9271 %   rmse: 24222  at train_T: 18  train time: 30.9556
bias: 146.6339 %   rmse: 232224  at train_T: 19  train time: 32.5082
bias: 14.2771 %   rmse: 24228  at train_T: 20  train time: 50.2770

"""








#可视化
# test_T=best_train_T+3
# train, test = y[:best_train_T * 24 * 4], y[best_train_T * 24 * 4:test_T * 24 * 4]
# time_begin = time.time()
# model = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 2, 1, best_train_T)).fit(disp=-1)  # 最佳参数（201，121）
# time_end = time.time()
# print('train time:', time_end - time_begin,' at train_T:',best_train_T)
# forecasts = model.forecast(test.shape[0])
# temp_bias=forecast_bias(best_train_T,forecasts,test)
# test_axis = np.arange(test.shape[0])
# plt.plot(test_axis, test, c='blue',label='test')
# plt.plot(test_axis, forecasts, c='green',label='forecasts')
# plt.legend()
# plt.show()

