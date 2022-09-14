from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

import pandas as pd
import numpy as np

from energy.dataset import dataset_buffered

class LD2011_2014(Dataset):

    def __init__(self, csv_file=r"dataset/LD2011_2014.csv", transform=None):
        self.transform = transform
        if csv_file not in dataset_buffered:
            dataset_buffered[csv_file] = pd.read_csv(csv_file)
        self.dataset = dataset_buffered[csv_file]

    def __len__(self):
        return self.dataset.shape[1]

    def __getitem__(self, index) -> T_co:
        pass