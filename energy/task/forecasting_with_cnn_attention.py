import numpy as np
import torch
import copy

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from dataset import London_11_14_random_select, construct_dataloader
from dataset.london_clean import London_11_14_set, createDataSet, createDataSetSingleFold
from helper.plot import plot_forecasting_weekly_for_comparison
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
PERIOD = 48
TIME_SIZE = 7 + 12
X_LENGTH = 30
Y_LENGTH = 7
EPOCH_STEP = 200
TOLERANCE = 40
LATITUDE_FACTOR = 1
LEARNING_RATE = 2e-3
MEANS_SCALE_FACTOR = 100
VARIANCES_SCALE_FACTOR = 10000
VARIANCES_DECAY = 2 * 1e-5
NOISE_STD_VARIANCE = 4

TASK_ID = "ForecastingWithCNNAttention"

bias_fn = nn.L1Loss()


def regression_display(model, sample):
    energy_x, energy_y, time_x, time_y = sample

    device = device_manager.device
    with torch.no_grad():
        pred = model(energy_x.to(device, dtype=torch.float32),
                     time_x.to(device, dtype=torch.float32),
                     time_y.to(device, dtype=torch.float32)).cpu().numpy()

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    means_cup, variances_cup = pred[:, :, 0].reshape(-1), pred[:, :, 1].reshape(-1)
    energy_x, energy_y = energy_x.reshape(-1).cpu().numpy(), energy_y.reshape(-1).cpu().numpy()

    axs[0].plot(range(energy_y.shape[0]), energy_y)
    axs[0].plot(range(energy_y.shape[0]), means_cup, color="red")
    axs[0].fill_between(range(energy_y.shape[0]), means_cup - LATITUDE_FACTOR * np.sqrt(variances_cup),
                        means_cup + LATITUDE_FACTOR * np.sqrt(variances_cup), facecolor='red', alpha=0.3)

    axs[1].plot(range(-energy_x.shape[0], energy_y.shape[0]), np.concatenate([energy_x, energy_y]))
    axs[1].plot(range(energy_y.shape[0]), means_cup, color="red")

    plt.show()


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


def mock_data_predicted(model):
    data = London_11_14_random_select(train_l=30, test_l=7, size=3000)
    dataloader = DataLoader(data, batch_size=64, shuffle=False)

    device = device_manager.device
    record = list()

    counter = 0
    with torch.no_grad():
        for (energy_x, energy_y, time_x, time_y) in dataloader:
            energy_x, time_x = energy_x.to(device, dtype=torch.float32), time_x.to(device, dtype=torch.float32)
            energy_y, time_y = energy_y.to(device, dtype=torch.float32), time_y.to(device, dtype=torch.float32)

            pred = model(energy_x, time_x, time_y)
            means = pred[:, :, 0]
            variances = pred[:, :, 1]

            date = data.df["DateTime"][23 * 48:]
            for load_observed, load_predicted, variance_predicted, load_future in zip(
                    energy_x.cpu().numpy(), means.cpu().numpy(),
                    variances.cpu().numpy(), energy_y.cpu().numpy()):
                record.append({
                    "Date_observed": list(date[counter * 48: counter * 48 + 7 * 48]),
                    "Load_observed": [float(v) for v in load_observed.reshape(-1)[23 * 48:]],
                    "Date_predicted": list(date[counter * 48 + 7 * 48: counter * 48 + 14 * 48]),
                    "Load_predicted": [float(v) for v in load_predicted.reshape(-1)],
                    "Variance_predicted": [float(v) for v in variance_predicted.reshape(-1)],
                    "Load_future":  [float(v) for v in load_future.reshape(-1)]
                })

                counter += 1

    return record


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

    # 需要更新新的画图模式
    plotter.plot_forecasting_random_samples_weekly(best_model, val.dataset, LATITUDE_FACTOR)
    plotter.plot_training_process()
    plotter.plot_sensitivity_curve_weekly(best_model, val.dataset)

    load_record = mock_data_predicted(predictor)
    return load_record


def create_dataset_multitask(k_flod=2):
    return createDataSet(k_flod=k_flod, train_l=X_LENGTH, label_l=Y_LENGTH, test_days=10,
                         test_continuous=3, size=3500, times=10)


def run_model_on_whole_data():
    predictor = load_task_model(TASK_ID, name="Date(2022-11-06 17-06-14).pth")
    predictor.eval()

    whole_dataset = London_11_14_random_select(train_l=X_LENGTH, test_l=Y_LENGTH, size=3500)
    dataloader = DataLoader(whole_dataset, shuffle=True)

    val_loop(dataloader, predictor, loss_function, tag="Whole Dataset")
    plot_forecasting_weekly_for_comparison(TASK_ID, predictor, whole_dataset, LATITUDE_FACTOR, 230)


RANDOM_SEED = 10001
torch.cuda.manual_seed(RANDOM_SEED)
if __name__ == "__main__":
    single_recorder = SingleTaskRecorder(TASK_ID)
    train_model(single_recorder, SingleTaskPlotter(single_recorder))
    # run_model_on_whole_data()