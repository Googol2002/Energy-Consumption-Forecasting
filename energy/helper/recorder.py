
"""
记录每次训练信息
"""
import os
from datetime import datetime, timedelta
from time import timezone

import pandas as pd
import torch

from helper import current_time_tag


class Recorder:

    def __init__(self, task_id, to_console=True, to_disk=True):
        # 记录时间标记
        self.date_tag = current_time_tag()

        self.task_id = task_id
        self.to_console = to_console
        self.to_disk = to_disk
        # 记录数据
        self.gradient_norms, self.train_losses, self.val_losses = [], [], []

    def __figure_directory_path(self):
        raise NotImplementedError()

    def __model_directory_path(self):
        raise NotImplementedError()

    def __stdout_path(self):
        raise NotImplementedError()

    def __performance_path(self):
        raise NotImplementedError()

    def __update_performance(self, train_accuracy, validation_accuracy,
                                    test_accuracy, model_name):
        path = self.__performance_path()
        performance = pd.DataFrame({"datetime": [self.date_tag], "train": [train_accuracy],
                                    "validation": [validation_accuracy], "test": [test_accuracy],
                                    "model_name": [model_name]}).astype(
            {"datetime": "str", "train": float, "validation": float, "test": float, "model_name": str})

        if os.path.exists(path):
            performance = pd.concat([performance, pd.read_csv(path)], ignore_index=True)

        performance.to_csv(path, index=False)

    def std_print(self, msg):
        if self.to_console:
            print(msg)

        if self.to_disk:
            with open(self.__stdout_path(), mode='a') as log_file:
                log_file.write(msg + "\n")

    def epoch_record(self, model=None, train_accuracy=None,
                     validation_accuracy=None, test_accuracy=None):

        model_name = "not saved"
        if model:
            path = self.__model_directory_path()
            model_name = r"Date({}).pth".format(self.date_tag)
            # torch.save(model.state_dict(), os.path.join(path, file_name))
            # 改为直接保存模型
            torch.save(model, os.path.join(path, model_name))
            self.std_print("Best Performing Model saved as {}".format(model_name))

        self.__update_performance(train_accuracy, validation_accuracy,
                                  test_accuracy, model_name)

    def training_record(self, train_loss, val_loss, gradient_norm=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)