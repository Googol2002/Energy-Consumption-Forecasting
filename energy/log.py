import torch
import time
import os

LOG_DIRECTORY = r"log"


def epoch_log(epoch, model_name, bias, model=None):
    now = int(time.time())
    # 转换为其他日期格式，如："%Y-%m-%d %H:%M:%S"
    time_arr = time.localtime(now)
    time_tag = time.strftime("%Y-%m-%d %H-%M-%S", time_arr)

    if model:
        folder = os.path.exists(os.path.join(LOG_DIRECTORY, model_name))
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(os.path.join(LOG_DIRECTORY, model_name))

        file_name = r"Epoch({})-Bias({:.2f})-Date({}).pth".format(epoch, bias, time_tag)
        torch.save(model.state_dict(), os.path.join(LOG_DIRECTORY, model_name, file_name))
        log_printf(model_name, "Model saved as {}".format(file_name))


start_time = int(time.time())
# 转换为其他日期格式，如："%Y-%m-%d %H:%M:%S"
start_arr = time.localtime(start_time)
start_tag = time.strftime("%Y-%m-%d %H-%M-%S", start_arr)


def log_printf(model_name, msg):

    folder = os.path.exists(os.path.join(LOG_DIRECTORY, model_name))
    if not folder:
        os.makedirs(os.path.join(LOG_DIRECTORY, model_name))

    with open(os.path.join(LOG_DIRECTORY, model_name, r"Test-Report-Date({}).txt".format(start_tag)), mode='a') as log_file:
        log_file.write(msg + "\n")

    print(msg)
