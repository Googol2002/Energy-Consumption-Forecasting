import os

from dataset.london_clean import *
import torch
import torch.multiprocessing as mp

from helper.device_manager import register_cuda_unit
from task.forecasting_with_cnn_attention import train_model as train


def process_runner(i, cuda_unit: str, dataset_path, pipe_in: mp.SimpleQueue):
    dataset = torch.load(dataset_path)
    if cuda_unit is not None:
        register_cuda_unit(cuda_unit)

    train(dataset=dataset, process_id=i)


class MultiTaskTrainer:

    def __init__(self, cuda_list):
        self.cuda_list = cuda_list

    def dispatch(self):
        processes = []
        for fold, cuda_unit in enumerate(self.cuda_list):
            pipe = mp.SimpleQueue()
            path = os.path.join("dataset", "10_flod_splits", "10_flod_split_{:0>2d}.pt".format(fold))
            ctx = mp.spawn(process_runner, (cuda_unit, path, pipe), join=False)
            processes.append(ctx)

        for p in processes:
            p.join()

        print("主线程结束了")


if __name__ == "__main__":
    trainer = MultiTaskTrainer(["cuda:1", "cuda:2"])
    trainer.dispatch()
