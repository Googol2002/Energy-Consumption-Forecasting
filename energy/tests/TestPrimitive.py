from multiprocessing.util import is_abstract_socket_namespace
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys
sys.path.append("/home/shihn/Oracle_shy")
sys.path.append("/home/shihn/Oracle_shy/energy")

from energy.dataset.london_clean import London_11_14_random_select

accuracy = []
for j in range(10):
    dataset = London_11_14_random_select(train_l=7, test_l=7, size=3000)
    accuracy_t = 0
    for i in range(len(dataset)):
        x, y, x1, y1 = dataset[i]
        # print("X:", x)
        # print("y:", y)
        accuracy_t += np.mean(1 - np.abs((x - y) / y))
    print("[{}/{}]".format(j, 10))

    accuracy_t /= len(dataset)
    accuracy_t *= 100
    accuracy.append(accuracy_t)

print(f"Accuracy: {accuracy} \nMean:{np.mean(accuracy)}")

# Graph

from helper.plot import primitive_plot_forecasting_random_samples_weekly
from helper import mute_log_plot


with mute_log_plot():
    primitive_plot_forecasting_random_samples_weekly("PrimitivelyForcastingNextWeek", London_11_14_random_select(train_l=7, test_l=7, size=3000), filename="Primitive")



