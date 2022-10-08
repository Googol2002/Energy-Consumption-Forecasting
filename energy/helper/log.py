from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import torch
import os
import re

LOG_DIRECTORY = r"log"

# 获取当前上海时间
SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)
utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)
shanghai_time = utc_time.astimezone(SHA_TZ)
# 转换为其他日期格式，如："%Y-%m-%d %H:%M:%S"
date_tag = shanghai_time.strftime("%Y-%m-%d %H-%M-%S")

# 用于静音log
is_muted = False
@contextmanager
def mute_log():
    global is_muted
    is_muted = True
    yield
    is_muted = False

def performance_log(task_id, msg, model=None):
    log_printf(task_id, msg)
    if model:
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(os.path.join(LOG_DIRECTORY, task_id)):
            os.makedirs(os.path.join(LOG_DIRECTORY, task_id))

        file_name = r"Date({}).pth".format(date_tag)
        torch.save(model.state_dict(), os.path.join(LOG_DIRECTORY, task_id, file_name))
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

