import os
from contextlib import contextmanager

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from helper import LOG_DIRECTORY, device_manager

plt.rcParams["font.sans-serif"] = ["SimHei"]    # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
CLIP = 10

class Plotter:

    def __init__(self, recorder, to_show=True, to_disk=True):
        self.recorder = recorder
        self.task_id = recorder.task_id
        self.to_disk = to_disk
        self.to_show = to_show

    def _save_figure(self, fig_name):
        raise NotImplementedError()

    @staticmethod
    def _plot(fig_name):
        def decorator(plot_function):
            def saver(self, *args, **kwargs):
                plot_function(self, *args, **kwargs)

                if self.to_disk:
                    self._save_figure(fig_name)
                if self.to_show:
                    plt.show()

            return saver
        return decorator

    @_plot("Examples_Weekly")
    def plot_forecasting_random_samples_weekly(self, model, dataset, factor, size=4):
        fig, axs = plt.subplots(size, 1, figsize=(16, size * 6))
        fig.tight_layout(pad=5.0)
        display_dataset = DataLoader(dataset, batch_size=size, shuffle=True)

        device = device_manager.device
        batch, (energy_x, energy_y, time_x, time_y) = next(iter(enumerate(display_dataset)))
        with torch.no_grad():
            pred = model(energy_x.to(device, dtype=torch.float32),
                         time_x.to(device, dtype=torch.float32),
                         time_y.to(device, dtype=torch.float32)).cpu().numpy()

        means_cup, variances_cup = pred[:, :, 0].reshape(size, -1), pred[:, :, 1].reshape(size, -1)
        energy_y = energy_y.reshape(size, -1).cpu().numpy()

        for i, (y, m, v) in enumerate(zip(energy_y, means_cup, variances_cup)):
            axs[i].plot(range(y.shape[0]), y)
            axs[i].plot(range(y.shape[0]), m, color="red")
            axs[i].fill_between(range(y.shape[0]), m - factor * np.sqrt(v),
                                m + factor * np.sqrt(v), facecolor='red', alpha=0.3)
            axs[i].title.set_text("Val Sample[{}]".format(i + 1))
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Energy Consumption")

    @_plot("TrainingProcess")
    def plot_training_process(self):
        fig, axs = plt.subplots(2, 1, figsize=(12, 12))

        train_loss = np.asarray(self.recorder.train_losses)
        val_loss = np.asarray(self.recorder.val_losses)
        gradient_norm = np.asarray(self.recorder.gradient_norms)
        indexes = np.asarray(range(len(train_loss)), dtype=int)

        clip = CLIP if len(indexes) > CLIP * 2 else 0

        axs[0].title.set_text("Training Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].plot(indexes[clip:], train_loss[clip:], label="Loss on Training Set")
        axs[0].plot(indexes[clip:], val_loss[clip:], label="Loss on Validation Set")
        axs[0].legend()

        axs[1].title.set_text("Training Gradient Norm")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Norm")
        axs[1].plot(indexes[clip:], np.log10(gradient_norm[clip:]), label="Log of Norm to the Base 10")
        axs[1].legend()

    @_plot("SensitivityCurve")
    def plot_sensitivity_curve_weekly(self, model,
                                      dataset, tolerance_range=None):
        tolerance_range = tolerance_range if tolerance_range else (0, 2)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        size, shape = len(dataset), dataset[0][1].shape

        @np.vectorize
        def evaluate(tolerance):
            within, utilization = 0, 0
            device = model.device
            with torch.no_grad():
                for (energy_x, energy_y, time_x, time_y) in dataloader:
                    energy_x, time_x = energy_x.to(device, dtype=torch.float32), time_x.to(device, dtype=torch.float32)
                    energy_y, time_y = energy_y.to(device, dtype=torch.float32), time_y.to(device, dtype=torch.float32)

                    pred = model(energy_x, time_x, time_y)

                    means = pred[:, :, 0]
                    variances = pred[:, :, 1]
                    within += torch.sum(energy_y <= means + tolerance * torch.sqrt(variances))
                    utilization += torch.sum(energy_y / (means + tolerance * torch.sqrt(variances)))

            return 100 * within.cpu() / (size * shape[0] * shape[1]), 100 * utilization.cpu() / (
                        size * shape[0] * shape[1])

        roc = evaluate(np.linspace(tolerance_range[0], tolerance_range[-1], 50))

        plt.plot(roc[0], roc[1])
        plt.grid()
        plt.xlabel("发电量覆盖用电量情况统计(%)")
        plt.ylabel("能源利用率(%)")
        plt.xticks(np.arange((min(roc[0]) // 5) * 5, 101, 5))
        plt.yticks(np.arange((min(roc[1]) // 5) * 5, 101, 5))
        plt.title("敏感度曲线(Sensitivity Curve)")


class SingleTaskPlotter(Plotter):

    def _save_figure(self, fig_name):
        path = os.path.join(LOG_DIRECTORY, self.task_id, "figure")
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(os.path.join(path, "{}-Date({}).png".
                                 format(fig_name, self.recorder.date_tag)),
                    dpi=300)


class MultiTaskPlotter(Plotter):

    def _save_figure(self, fig_name):
        path = os.path.join(LOG_DIRECTORY, self.task_id, self.recorder.date_tag, "figure",
                            "processor({})".format(self.recorder.process_id))
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(os.path.join(path, "{}-Date({}).png".
                                 format(fig_name, self.recorder.date_tag)),
                    dpi=300)