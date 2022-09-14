import pytest

from energy.dataset import LD2011_2014

def test_ld2011_2014():
    LENGTH = 1000
    dataset = LD2011_2014(length=LENGTH, csv_file=r"D:\Workspace\Energy-Consumption-Forecasting\dataset\LD2011_2014.csv")
    for sample in dataset:
        assert(sample[0].shape[0] == LENGTH)