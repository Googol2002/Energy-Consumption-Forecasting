import json
import os
from statistics import mean

from dataset.london_clean import *
import torch
import torch.multiprocessing as mp

from helper import current_time_tag
from helper.device_manager import register_cuda_unit
from helper.plotter import MultiTaskPlotter
from helper.recorder import MultiTaskRecorder
from task.forecasting_with_cnn_attention import train_model as train

TASK_ID = "MULTI_TASK"

def process_runner(fold, cuda_unit: str, dataset_path, date_tag):
    dataset = torch.load(dataset_path)
    if cuda_unit is not None:
        register_cuda_unit(cuda_unit)

    recorder = MultiTaskRecorder(TASK_ID, fold, to_console=1, date_tag=date_tag)
    plotter = MultiTaskPlotter(recorder, to_show=False)
    train(recorder, plotter, dataset=dataset, process_id=fold)

    return recorder.to_json()


class MultiTaskTrainer:

    def __init__(self, cuda_list):
        self.cuda_list = cuda_list
        self.date_tag = current_time_tag()
        self.results= []

    def dispatch(self, out_file=None):
        mp.set_start_method("spawn")
        tasks = []
        with mp.Pool(processes=len(self.cuda_list)) as pool:
            for fold, cuda_unit in enumerate(self.cuda_list):
                path = os.path.join("dataset", "10_flod_splits", "10_flod_split_{:0>2d}.pt".format(fold))
                task = pool.apply_async(process_runner, (fold, cuda_unit, path, self.date_tag))
                tasks.append(task)

            for task in tasks:
                self.results.append(json.loads(task.get()))

        K_folds_result = json.dumps({
            "Task ID": TASK_ID,
            "10-K Accuracy": mean([r["Test Accuracy"] for r in self.results]),
            "records": self.results
        })
        if out_file:
            with open(out_file, 'w') as f:
                f.write(K_folds_result)

        return K_folds_result


if __name__ == "__main__":
    trainer = MultiTaskTrainer(["cuda:1", "cuda:2", "cuda:3", "cuda:0"])
    trainer.dispatch(out_file=r"out.json")
