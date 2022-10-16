import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys
sys.path.append("/home/shihn/Oracle_shy")
sys.path.append("/home/shihn/Oracle_shy/energy")

from energy.dataset.london_clean import London_11_14_random_select


dataset = London_11_14_random_select(train_l=7, test_l=7, size=3000)

accuracy = []
for j in range(100):
    accuracy_t = 0
    for i in range(len(dataset)):
        x, y, x1, y1 = dataset[i]
        # print("X:", x)
        # print("y:", y)
        accuracy_t += np.mean(1 - (x - y) / y)

    accuracy_t /= len(dataset)
    accuracy_t *= 100
    accuracy.append(accuracy_t)

print(f"Accuracy: {accuracy} Mean:{np.mean(accuracy)}")
