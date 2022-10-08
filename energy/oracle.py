import argparse

# 这里将提供一个面向控制台的简单操作界面
import importlib
import os

parser = argparse.ArgumentParser(description='The Oracle是一个电力负载预测模型.')
parser.add_argument('--train', action='store_true', help='训练模型')
parser.add_argument('--test', action='store_true', help='测试模型')
parser.add_argument('bar', metavar='forecasting_next_day', type=str, nargs='+',
                    help='需要训练任务的名称，可在task文件夹下获得.')

if __name__ == "__main__":
    args = parser.parse_args()
    os.chdir(r"../")
    for task_name in args.bar:
        task = importlib.import_module('task.' + task_name)
        if args.train:
            task.train_model()
        if args.test:
            task.test_model()
