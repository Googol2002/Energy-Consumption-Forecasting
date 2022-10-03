from datetime import datetime, timedelta, timezone

import torch
import time
import os

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


def performance_log(model_name, msg, model=None):
    log_printf(model_name, msg)
    if model:
        folder = os.path.exists(os.path.join(LOG_DIRECTORY, model_name))
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os.path.join(LOG_DIRECTORY, model_name))

        file_name = r"Date({}).pth".format(date_tag)
        torch.save(model.state_dict(), os.path.join(LOG_DIRECTORY, model_name, file_name))
        log_printf(model_name, "Best Performing Model saved as {}".format(file_name))


def log_printf(model_name, msg):

    folder = os.path.exists(os.path.join(LOG_DIRECTORY, model_name))
    if not folder:
        os.makedirs(os.path.join(LOG_DIRECTORY, model_name))

    with open(os.path.join(LOG_DIRECTORY, model_name, r"Test-Report-Date({}).txt".format(date_tag)), mode='a') as log_file:
        log_file.write(msg + "\n")

    print(msg)
