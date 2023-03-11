
"""
记录每次训练信息
"""
import os

import pandas as pd
import torch
import json

from helper import current_time_tag, LOG_DIRECTORY


class Recorder:

    """
    只有level大于等于to_console和to_disk等级的log才会被输出到对应位置
    """
    def __init__(self, task_id, to_console=0, to_disk=0):
        # 记录时间标记
        self.date_tag = current_time_tag()

        self.task_id = task_id
        self.to_console = to_console
        self.to_disk = to_disk
        # 记录训练数据，需要被保存在JSON中
        self.gradient_norms, self.train_losses, self.val_losses = [], [], []
        # 记录Summary数据，需要被保存在JSON中
        self.train_accuracy, self.validation_accuracy,\
            self.test_accuracy, self.model_name = None, None, None, None

    def std_print(self, msg, level=0):
        if level >= self.to_console:
            print(msg)

        if level >= self.to_disk:
            with open(self._stdout_path(), mode='a') as log_file:
                log_file.write(msg + "\n")

    def summary_record(self, model=None, train_accuracy=None,
                       validation_accuracy=None, test_accuracy=None):
        model_name = "not saved"
        if model:
            path = self._model_directory_path()
            model_name = r"Date({}).pth".format(self.date_tag)
            # torch.save(model.state_dict(), os.path.join(path, file_name))
            # 改为直接保存模型
            torch.save(model, os.path.join(path, model_name))
            self.std_print("Best Performing Model saved as {}".format(model_name))

        # 记录结果信息
        self.train_accuracy, self.validation_accuracy, self.test_accuracy,\
            self.model_name = train_accuracy, validation_accuracy, test_accuracy, model_name
        self._record_performance(train_accuracy, validation_accuracy,
                                 test_accuracy, model_name)

    def _model_directory_path(self):
        raise NotImplementedError()

    def _stdout_path(self):
        raise NotImplementedError()

    def _record_performance(self, train_accuracy, validation_accuracy,
                            test_accuracy, model_name):
        raise NotImplementedError()

    def training_record(self, train_loss, val_loss, gradient_norm=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)

    def to_json(self):
        return json.dumps(self._to_dictionary())

    def _to_dictionary(self):
        return {"Task ID": self.task_id,
                "Date_Tag": self.date_tag,
                "Gradient Norms": self.gradient_norms,
                "Train Losses": self.train_losses,
                "Val Losses": self.val_losses,
                "Train Accuracy": self.train_accuracy,
                "Validation Accuracy": self.validation_accuracy,
                "Test Accuracy": self.test_accuracy,
                "Model Name": self.model_name
                }


class SingleTaskRecorder(Recorder):

    def _record_performance(self, train_accuracy, validation_accuracy, test_accuracy, model_name):
        path = os.path.join(LOG_DIRECTORY, self.task_id, "performance.csv")
        performance = pd.DataFrame({"datetime": [self.date_tag], "train": [train_accuracy],
                                    "validation": [validation_accuracy], "test": [test_accuracy],
                                    "model_name": [model_name]}).astype(
            {"datetime": "str", "train": float, "validation": float, "test": float, "model_name": str})

        if os.path.exists(path):
            performance = pd.concat([performance, pd.read_csv(path)], ignore_index=True)

        performance.to_csv(path, index=False)

    def __init__(self, task_id, to_console=True, to_disk=True):
        super().__init__(task_id, to_console=to_console, to_disk=to_disk)

    def _model_directory_path(self):
        path = os.path.join(LOG_DIRECTORY, self.task_id, "model")
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def _stdout_path(self):
        folder = os.path.join(LOG_DIRECTORY, self.task_id, "output")
        path = os.path.join(folder, r"Test-Report-Date({}).txt".format(self.date_tag))
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(folder):
            os.makedirs(folder)

        return path


class MultiTaskRecorder(Recorder):

    def __init__(self, task_id, process_id, to_console=True, to_disk=True, date_tag=None):
        super().__init__(task_id, to_console=to_console, to_disk=to_disk)
        self.process_id = process_id
        if date_tag:
            self.date_tag = date_tag
        # named Train, Validation, Test
        self.accuracy = (0, 0, 0)

    def _model_directory_path(self):
        path = os.path.join(LOG_DIRECTORY, self.task_id, self.date_tag,
                            "model", "processor({})".format(self.process_id))
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def _stdout_path(self):
        folder = os.path.join(LOG_DIRECTORY, self.task_id, self.date_tag,
                              "output", "processor({})".format(self.process_id))
        path = os.path.join(folder, r"Test-Report-Date({}).txt".format(self.date_tag))
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(folder):
            os.makedirs(folder)

        return path

    def _to_dictionary(self):
        dictionary = super()._to_dictionary()
        dictionary["Process ID"] = self.process_id

        return dictionary

    """
    多任务训练由父进程记录表现
    """
    def _record_performance(self, train_accuracy, validation_accuracy,
                            test_accuracy, model_name):
        pass
