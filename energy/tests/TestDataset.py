import pytest
import numpy as np

from energy.dataset import LD2011_2014

def test_ld2011_2014_access():
    LENGTH = 1000
    dataset = LD2011_2014(length=LENGTH, csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv")
    for index, sample in enumerate(dataset):
        x = sample[0]
        y = sample[1]
        assert(x.shape[0] == LENGTH)
        assert((x == dataset[index][0]).all())
        assert(y == dataset[index][1])