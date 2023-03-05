import os

from dataset.london_clean import *
import torch
import torch.multiprocessing as mp

from task.forecasting_with_cnn_attention import train_model as train


def process_runner(cuda_unit: str, remaining_cuda: mp.Semaphore, dataset_path,  pipe_in: mp.SimpleQueue):
    remaining_cuda.acquire()
    dataset = torch.load(dataset_path)
    train(dataset=dataset, cuda_unit=cuda_unit)
    remaining_cuda.release()


class MultiTaskTrainer:

    def __init__(self, cuda_list):
        self.cuda_list = cuda_list

    def dispatch(self):
        remaining_cuda = mp.Semaphore(len(self.cuda_list))
        processes = []
        for fold, cuda_unit in enumerate(self.cuda_list):
            pipe = mp.SimpleQueue()
            path = os.path.join("dataset", "10_flod_splits", "10_flod_split_{:0>2d}.pt".format(fold))
            p = mp.Process(target=process_runner,
                           args=(cuda_unit, remaining_cuda, path, pipe))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("主线程结束了")


if __name__ == "__main__":
    mp.set_start_method('spawn')

    trainer = MultiTaskTrainer(["cuda:0", "cuda:1"])
    trainer.dispatch()
