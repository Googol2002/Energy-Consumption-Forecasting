import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from energy.dataset import LD2011_2014_summary
# Load/split your data
dataset = LD2011_2014_summary(0)
y = dataset.dataset
train, test = y[:28*24*4], y[28*24*4:35*24*4]

# Fit your model
model = pm.auto_arima(train, seasonal=True, m=train.shape[0] // (4 * 24))


# 模型导出与载入
import joblib
joblib.dump(model,'auto_arima.pkl')
# model = joblib.load(model,'auto_arima.pkl')

# make your forecasts
# predict N steps into the future
forecasts = model.predict(test.shape[0])  

# Visualize the forecasts (blue=train, green=forecasts)
test_axis = np.arange(test.shape[0])
plt.plot(test_axis, test, c='blue')
plt.plot(test_axis, forecasts, c='green')
plt.savefig("ARIMA.png")
# plt.show()
