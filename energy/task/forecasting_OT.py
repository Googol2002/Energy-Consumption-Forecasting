import numpy as np
import torch
import copy

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from dataset import construct_dataloader
from dataset.ETT_data import Dataset_ETT_hour
from helper.plotter import SingleTaskPlotter
from helper.recorder import SingleTaskRecorder
from model import check_gradient_norm_L2

from model.AdvancedModel import CNN_Attention_Model
from model.PeriodicalModel import customize_loss

from helper.log import load_task_model

import helper.device_manager as device_manager

GRADIENT_NORM = 100
WEIGHT_DECAY = 0.01
BATCH_SIZE = 128
HIDDEN_SIZE = 256
KERNEL_SIZE = 7
PERIOD = 24
TIME_SIZE = 7 + 12
X_LENGTH = 16
Y_LENGTH = 4
EPOCH_STEP = 200
TOLERANCE = 40
LATITUDE_FACTOR = 1
LEARNING_RATE = 2e-3
# MEANS_SCALE_FACTOR = 100
# VARIANCES_SCALE_FACTOR = 10000
VARIANCES_DECAY = 2 * 1e-5
NOISE_STD_VARIANCE = 0.01

TASK_ID = "Forecasting_OT"


def val_loop(dataloader, model, loss_fn, recorder, tag="Val"):
    size = len(dataloader.dataset)
    val_loss, accuracy, within, utilization = 0, 0, 0, 0

    device = device_manager.device
    with torch.no_grad():
        for (energy_x, energy_y, time_x, time_y) in dataloader:
            energy_x, time_x = energy_x.to(device, dtype=torch.float32), time_x.to(device, dtype=torch.float32)
            energy_y, time_y = energy_y.to(device, dtype=torch.float32), time_y.to(device, dtype=torch.float32)

            pred = model(energy_x, time_x, time_y)
            val_loss += loss_fn(pred, energy_y).item()

            means = pred[:, :, 0]
            variances = pred[:, :, 1]
            accuracy += torch.sum(torch.abs(means - energy_y) / energy_y).item()
            within += torch.sum(energy_y <= means + LATITUDE_FACTOR * torch.sqrt(variances))
            utilization += torch.sum(energy_y / (means + LATITUDE_FACTOR * torch.sqrt(variances)))

    val_loss /= (size * PERIOD)
    recorder.std_print(tag + " " + f"Error: \n Accuracy: {100 - (100 * accuracy / (size * PERIOD * Y_LENGTH)):>0.3f}%,"
                                   f" Avg loss: {val_loss:>8f}")
    recorder.std_print(f" Within the Power Generation: {(100 * within / (size * PERIOD * Y_LENGTH)):>0.3f}%")
    recorder.std_print(f" Utilization Rate:  {(100 * utilization / (size * PERIOD * Y_LENGTH)):>0.3f}%\n")

    return val_loss, 1 - accuracy / (size * PERIOD * Y_LENGTH)


def train_loop(dataloader, model, loss_fn, optimizer, recorder):
    size = len(dataloader.dataset)
    total_loss, gradient_norm = 0, 0

    device = device_manager.device
    for batch, (energy_x, energy_y, time_x, time_y) in enumerate(dataloader):
        energy_x, time_x = energy_x.to(device, dtype=torch.float32), time_x.to(device, dtype=torch.float32)
        energy_y, time_y = energy_y.to(device, dtype=torch.float32), time_y.to(device, dtype=torch.float32)
        # Gaussian Noise 对抗过拟合
        energy_x += (torch.randn(energy_x.shape, device=device) * NOISE_STD_VARIANCE)

        # Compute prediction and loss
        pred = model(energy_x, time_x, time_y)
        loss = loss_fn(pred, energy_y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # 记录梯度大小
        gradient_norm += check_gradient_norm_L2(model)
        # clip
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
        optimizer.step()

        total_loss += loss.item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(energy_x)
            recorder.std_print(f"loss: {loss:>7f} Avg loss: {loss / (energy_x.shape[0] * PERIOD) :>7f}  [{current:>5d}/{size:>5d}]")

    return total_loss / (PERIOD * size), gradient_norm ** 0.5


loss_function = customize_loss(VARIANCES_DECAY)

def train_model(recorder, plotter, datasets=None, process_id=None):
    global TASK_ID
    TASK_ID = "{}({})".format(TASK_ID, process_id)

    train, val, test = (DataLoader(d, batch_size=BATCH_SIZE) for d in datasets)

    predictor = CNN_Attention_Model(input_size=PERIOD, hidden_size=HIDDEN_SIZE, num_layers=1,
                                    output_size=PERIOD, batch_size=BATCH_SIZE, period=PERIOD,
                                    attention_size=X_LENGTH,    # 设置了固定Attention的长度
                                    time_size=TIME_SIZE, means=1, kernel_size=KERNEL_SIZE,
                                    means_scale_factor=1,
                                    variances_scale_factor=1,
                                    mlp_sizes=[HIDDEN_SIZE * 2 + TIME_SIZE, HIDDEN_SIZE, 128, 128, 64, PERIOD])

    recorder.std_print(TASK_ID + ' model:' + "\n" + str(predictor))
    adam = torch.optim.Adam(predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_model = None
    min_val_loss = 50000000000
    tolerance = 0
    for epoch in range(EPOCH_STEP):
        recorder.std_print("========EPOCH {}========\n".format(epoch))
        train_loss, norm = train_loop(train, predictor, loss_function, adam, recorder)
        validation_loss, bias = val_loop(val, predictor, loss_function, recorder)
        tolerance += 1
        if min_val_loss > validation_loss > 0:
            min_val_loss = validation_loss
            best_model = copy.deepcopy(predictor)
            tolerance = 0
        if tolerance > TOLERANCE:
            if process_id is None:
                recorder.std_print(TASK_ID, "Early stopped at epoch {}.\n".format(epoch))
            else:
                recorder.std_print("Process({}): Early stopped at {:>3d}/{:>3d}".format(process_id, epoch, EPOCH_STEP), level=1)
            break
        recorder.training_record(train_loss, validation_loss, gradient_norm=norm)

        if process_id is not None and epoch % 5 == 0:
            recorder.std_print("Process({}):{:>3d}/{:>3d}".format(process_id, epoch, EPOCH_STEP), level=1)

    best_model.lstm.flatten_parameters()
    recorder.std_print("========Best Performance========\n")
    _, train_accuracy = val_loop(train, best_model, loss_function, recorder, tag="Train")
    _, validation_accuracy = val_loop(val, best_model, loss_function, recorder, tag="Val")
    _, test_accuracy = val_loop(test, best_model, loss_function, recorder, tag="Test")
    recorder.summary_record(model=best_model, train_accuracy=train_accuracy,
                            validation_accuracy=validation_accuracy, test_accuracy=test_accuracy)

    recorder.std_print("Process({}) Training Completed!".format(process_id), level=1)

    # 画图测试
    # display_dataset = DataLoader(val.dataset, batch_size=1, shuffle=True)
    # regression_display(best_model, next(iter(display_dataset)))

    # 需要更新新的画图模式
    plotter.plot_forecasting_random_samples_weekly(best_model, val.dataset, LATITUDE_FACTOR)
    plotter.plot_training_process()
    plotter.plot_sensitivity_curve_weekly(best_model, val.dataset)


RANDOM_SEED = 10001
torch.cuda.manual_seed(RANDOM_SEED)
if __name__ == "__main__":
    single_recorder = SingleTaskRecorder(TASK_ID)
    datasets = [Dataset_ETT_hour(root_path='dataset/ETT-small', timeenc=0, scale=True,
                                 inverse=False,  features='S', target='OT', freq='h',
                                 flag='train', data_path='ETTh2.csv',
                                 size=[24 * 4 * 4, 0, 24 * 4], window=24)
                for f in ['train', 'val', 'test']]
    train_model(single_recorder, SingleTaskPlotter(single_recorder), datasets=datasets)
