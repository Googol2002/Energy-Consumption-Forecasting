import numpy as np
import torch
import copy

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from dataset import London_11_14_random_select, construct_dataloader
from dataset.london_clean import London_11_14_set, createDataSet, createDataSetSingleFold
from helper.plot import plot_forecasting_random_samples_weekly, plot_training_process, plot_sensitivity_curve_weekly
from helper.plotter import SingleTaskPlotter
from helper.recorder import SingleTaskRecorder

from model.AdvancedModel import CNN_Attention_Model
from model.PeriodicalModel import customize_loss

from helper.log import load_task_model

import helper.device_manager as device_manager

GRADIENT_NORM = 100
WEIGHT_DECAY = 0.01
BATCH_SIZE = 128
HIDDEN_SIZE = 256
KERNEL_SIZE = 7
PERIOD = 48
TIME_SIZE = 7 + 12
X_LENGTH = 30
Y_LENGTH = 7
EPOCH_STEP = 5
TOLERANCE = 40
LATITUDE_FACTOR = 1
LEARNING_RATE = 2e-3
MEANS_SCALE_FACTOR = 100
VARIANCES_SCALE_FACTOR = 10000
VARIANCES_DECAY = 2 * 1e-5
NOISE_STD_VARIANCE = 4

TASK_ID = "ForecastingWithCNNAttention"

bias_fn = nn.L1Loss()


def check_gradient_norm_L2(model):
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm

    return total_norm


# def regression_display(model, sample):
#     energy_x, energy_y, time_x, time_y = sample
#
#     device = device_manager.device
#     with torch.no_grad():
#         pred = model(energy_x.to(device, dtype=torch.float32),
#                      time_x.to(device, dtype=torch.float32),
#                      time_y.to(device, dtype=torch.float32)).cpu().numpy()
#
#     fig, axs = plt.subplots(2, 1, figsize=(12, 12))
#
#     means_cup, variances_cup = pred[:, :, 0].reshape(-1), pred[:, :, 1].reshape(-1)
#     energy_x, energy_y = energy_x.reshape(-1).cpu().numpy(), energy_y.reshape(-1).cpu().numpy()
#
#     axs[0].plot(range(energy_y.shape[0]), energy_y)
#     axs[0].plot(range(energy_y.shape[0]), means_cup, color="red")
#     axs[0].fill_between(range(energy_y.shape[0]), means_cup - LATITUDE_FACTOR * np.sqrt(variances_cup),
#                         means_cup + LATITUDE_FACTOR * np.sqrt(variances_cup), facecolor='red', alpha=0.3)
#
#     axs[1].plot(range(-energy_x.shape[0], energy_y.shape[0]), np.concatenate([energy_x, energy_y]))
#     axs[1].plot(range(energy_y.shape[0]), means_cup, color="red")
#
#     plt.show()


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
    recorder.std_print(tag + " " + f"Error: \n Accuracy: {100 - (100 * accuracy / (size * PERIOD * Y_LENGTH)):>0.3f}%, Avg loss: {val_loss:>8f}")
    recorder.std_print(f" Within the Power Generation: {(100 * within / (size * PERIOD * Y_LENGTH)):>0.3f}%")
    recorder.std_print(f" Utilization Rate:  {(100 * utilization / (size * PERIOD * Y_LENGTH)):>0.3f}%\n")

    return val_loss, 1 - accuracy / (size * PERIOD * Y_LENGTH)


def train_loop(dataloader, model, loss_fn, optimizer):
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
            print(f"loss: {loss:>7f} Avg loss: {loss / (energy_x.shape[0] * PERIOD) :>7f}  [{current:>5d}/{size:>5d}]")

    return total_loss / (PERIOD * size), gradient_norm ** 0.5


loss_function = customize_loss(VARIANCES_DECAY)

def train_model(recorder, plotter, dataset=None, process_id=None):
    global TASK_ID
    TASK_ID = "{}({})".format(TASK_ID, process_id)

    train_set, val_and_test_set, energy_expectations, energy_variances = dataset if dataset \
        else createDataSetSingleFold(train_l=X_LENGTH, label_l=Y_LENGTH, test_days=10,
                                     test_continuous=3, size=3500, times=10)

    val, test = construct_dataloader(val_and_test_set, train_ratio=0.5,
                                     validation_ratio=0.5, test_ratio=0,
                                     batch_size=BATCH_SIZE)
    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # val = DataLoader(val_and_test_set, batch_size=BATCH_SIZE)
    print(len(train_set), len(val_and_test_set))

    predictor = CNN_Attention_Model(input_size=PERIOD, hidden_size=HIDDEN_SIZE, num_layers=1,
                                    output_size=PERIOD, batch_size=BATCH_SIZE, period=PERIOD,
                                    attention_size=X_LENGTH,    # 设置了固定Attention的长度
                                    time_size=TIME_SIZE, means=energy_expectations, kernel_size=KERNEL_SIZE,
                                    means_scale_factor=MEANS_SCALE_FACTOR,
                                    variances_scale_factor=VARIANCES_SCALE_FACTOR,
                                    mlp_sizes=[HIDDEN_SIZE * 2 + TIME_SIZE, HIDDEN_SIZE, 128, 128, 64, PERIOD])

    recorder.std_print(TASK_ID + ' model:' + "\n" + str(predictor))
    adam = torch.optim.Adam(predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_model = None
    min_val_loss = 50000000000
    tolerance = 0
    print('train_sum=', len(train))

    for epoch in range(EPOCH_STEP):
        print("========EPOCH {}========\n".format(epoch))
        train_loss, norm = train_loop(train, predictor, loss_function, adam)
        validation_loss, bias = val_loop(val, predictor, loss_function, recorder)
        tolerance += 1
        if min_val_loss > validation_loss > 0:
            min_val_loss = validation_loss
            best_model = copy.deepcopy(predictor)
            tolerance = 0
        if tolerance > TOLERANCE:
            recorder.std_print(TASK_ID, "Early stopped at epoch {}.\n".format(epoch))
            break
        recorder.training_record(train_loss, validation_loss, gradient_norm=norm)

    best_model.lstm.flatten_parameters()
    recorder.std_print("========Best Performance========\n")
    _, train_accuracy = val_loop(train, best_model, loss_function, recorder, tag="Train")
    _, validation_accuracy = val_loop(val, best_model, loss_function, recorder, tag="Val")
    _, test_accuracy = val_loop(test, best_model, loss_function, recorder, tag="Test")
    recorder.epoch_record(model=best_model, train_accuracy=train_accuracy,
                          validation_accuracy=validation_accuracy, test_accuracy=test_accuracy)
    # 画图测试
    # display_dataset = DataLoader(val.dataset, batch_size=1, shuffle=True)
    # regression_display(best_model, next(iter(display_dataset)))

    # 需要更新新的画图模式
    plotter.plot_forecasting_random_samples_weekly(best_model, val.dataset, LATITUDE_FACTOR)
    plotter.plot_training_process()
    plotter.plot_sensitivity_curve_weekly(best_model, val.dataset)

    # print("正向Softmax:", best_model.softmax(best_model.attention[0]))
    # print("负向Softmax:", best_model.softmax(best_model.attention[1]))


def create_dataset_multitask(k_flod=2):
    return createDataSet(k_flod=k_flod, train_l=X_LENGTH, label_l=Y_LENGTH, test_days=10,
                         test_continuous=3, size=3500, times=10)


def test_model_on_whole_data():
    predictor = load_task_model(TASK_ID, name="Date(2022-11-06 17-06-14).pth")
    predictor.eval()

    # whole_dataset = London_11_14_random_select(train_l=X_LENGTH, test_l=Y_LENGTH, size=3500)
    # dataloader = DataLoader(whole_dataset, shuffle=True)

    # val_loop(dataloader, predictor, loss_function, tag="Whole Dataset")
    # plot_forecasting_weekly_for_comparison(TASK_ID, predictor, whole_dataset, LATITUDE_FACTOR, 230)


RANDOM_SEED = 10001
torch.cuda.manual_seed(RANDOM_SEED)
if __name__ == "__main__":
    recorder = SingleTaskRecorder(TASK_ID)
    train_model(recorder, SingleTaskPlotter(recorder))
