import contextlib
from datetime import datetime

import pandas as pd
import torch
import os
import re

from helper import LOG_DIRECTORY, is_muted, training_recoder, TrainingProcess, current_time_tag

date_tag = current_time_tag()

def _model_directory_path(task_id):
    path = os.path.join(LOG_DIRECTORY, task_id, "model")
    # 判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def _output_directory_path(task_id):
    path = os.path.join(LOG_DIRECTORY, task_id, "output")
    # 判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def _append_performance_recode(task_id, train_accuracy, validation_accuracy, test_accuracy, model_name):
    path = os.path.join(LOG_DIRECTORY, task_id, "performance.csv")
    performance = pd.DataFrame({"datetime": [date_tag], "train": [train_accuracy],
                                "validation": [validation_accuracy], "test": [test_accuracy],
                                "model_name": [model_name]}).astype(
        {"datetime": "str", "train": float, "validation": float, "test": float, "model_name": str})

    if os.path.exists(path):
        performance = pd.concat([performance, pd.read_csv(path)], ignore_index=True)

    performance.to_csv(path, index=False)

def performance_log(task_id, msg=None, model=None, train_accuracy=None,
                    validation_accuracy=None, test_accuracy=None):

    if msg:
        log_printf(task_id, msg)

    model_name = "not saved"

    if model:
        path = _model_directory_path(task_id)

        model_name = r"Date({}).pth".format(date_tag)
        # torch.save(model.state_dict(), os.path.join(path, file_name))
        # 改为直接保存模型
        torch.save(model, os.path.join(path, model_name))
        log_printf(task_id, "Best Performing Model saved as {}".format(model_name))

    _append_performance_recode(task_id, train_accuracy, validation_accuracy, test_accuracy, model_name)


def log_printf(task_id, msg):
    print(msg)

    if is_muted:
        return

    with open(os.path.join(_output_directory_path(task_id), r"Test-Report-Date({}).txt".format(date_tag)),
              mode='a') as log_file:
        log_file.write(msg + "\n")


regex_date = re.compile(r"Date\(([\d\- ]+?)\)\.pth")
'''
:parameter name: 如果为空的话，会默认加载最近生成的模型.
'''
def load_task_model(task_id, name=None):
    if not name:
        model_names = [os.fsdecode(file) for file in os.listdir(os.path.join(LOG_DIRECTORY, task_id))
                       if os.fsdecode(file).endswith(".pth")]
        if not model_names:
            raise FileNotFoundError("测试或加载模型前请先训练.")
        model_dates = [datetime.strptime(regex_date.findall(name)[0], "%Y-%m-%d %H-%M-%S").timestamp()
                       for name in model_names]
        name = max(zip(model_dates, model_names), key=lambda t: t[0])[1]

    path = os.path.join(_model_directory_path(task_id), name)
    model_data = torch.load(path)
    print("Model loaded from {} :".format(path))
    # for param_tensor in state_dict:
    #     print(param_tensor, "\t", state_dict[param_tensor].size())

    return model_data


def record_training_process(task_id, train_loss, val_loss, gradient_norm=None):
    if task_id not in training_recoder:
        training_recoder[task_id] = TrainingProcess([], [], [])
    recoder = training_recoder[task_id]
    recoder.train_loss.append(train_loss)
    recoder.val_loss.append(val_loss)
    if gradient_norm is not None:
        recoder.gradient_norm.append(gradient_norm)
