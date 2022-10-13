from datetime import datetime, timedelta, timezone

import torch
import os
import re

from helper import LOG_DIRECTORY, is_muted, training_recoder, TrainingProcess

# 获取当前上海时间
SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)
utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)
shanghai_time = utc_time.astimezone(SHA_TZ)
# 转换为其他日期格式，如："%Y-%m-%d %H:%M:%S"
date_tag = shanghai_time.strftime("%Y-%m-%d %H-%M-%S")


def _model_directory_path(task_id):
    return os.path.join(LOG_DIRECTORY, task_id, "model")


def performance_log(task_id, msg, model=None):
    log_printf(task_id, msg)
    if model:
        path = _model_directory_path(task_id)
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = r"Date({}).pth".format(date_tag)
        torch.save(model.state_dict(), os.path.join(path, file_name))
        log_printf(task_id, "Best Performing Model saved as {}".format(file_name))


def log_printf(task_id, msg):
    print(msg)

    if is_muted:
        return

    if not os.path.exists(os.path.join(LOG_DIRECTORY, task_id)):
        os.makedirs(os.path.join(LOG_DIRECTORY, task_id))

    with open(os.path.join(LOG_DIRECTORY, task_id, r"Test-Report-Date({}).txt".format(date_tag)), mode='a') as log_file:
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

    path = os.path.join(LOG_DIRECTORY, task_id, name)
    state_dict = torch.load(path)
    print("Model loaded from {} :".format(path))
    # for param_tensor in state_dict:
    #     print(param_tensor, "\t", state_dict[param_tensor].size())

    return state_dict


def record_training_process(task_id, train_loss, val_loss, gradient_norm=None):
    if task_id not in training_recoder:
        training_recoder[task_id] = TrainingProcess([], [], [])
    recoder = training_recoder[task_id]
    recoder.train_loss.append(train_loss)
    recoder.val_loss.append(val_loss)
    if gradient_norm is not None:
        recoder.gradient_norm.append(gradient_norm)
