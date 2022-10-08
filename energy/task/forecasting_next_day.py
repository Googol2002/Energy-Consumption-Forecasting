import numpy as np
import torch
import copy

from matplotlib import pyplot as plt
from torch import nn

from dataset import construct_dataloader, LD2011_2014_summary_by_day
from helper.plot import plot_forecasting_random_samples
from model.PeriodicalModel import PeriodicalModel, customized_loss

from helper import log_printf, performance_log, load_task_model, mute_log

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GRADIENT_NORM = 10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
HIDDEN_SIZE = 512
PERIOD = 96
LENGTH = 30
EPOCH_STEP = 300
TOLERANCE = 20
LATITUDE_FACTOR = 1

TASK_ID = "ForecastingNextDay"

bias_fn = nn.L1Loss()

def check_gradient_norm(model):
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print("Gradient norm: {:>7f}".format(total_norm))


def regression_display(model, sample):
    with torch.no_grad():
        pred = model(torch.unsqueeze(torch.Tensor(sample[0]).to(device), 0)).cpu().numpy()
    x = sample[0].reshape(-1)
    y = sample[1].reshape(-1)

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    means_cup = pred[:, 0].reshape(-1)
    variances_cup = pred[:, 1].reshape(-1)

    axs[0].plot(range(y.shape[0]), y)
    axs[0].plot(range(y.shape[0]), means_cup, color="red")
    axs[0].fill_between(range(y.shape[0]), means_cup - LATITUDE_FACTOR * np.sqrt(variances_cup),
                        means_cup + LATITUDE_FACTOR * np.sqrt(variances_cup), facecolor='red', alpha=0.3)

    axs[1].plot(range(-x.shape[0], y.shape[0]), np.concatenate([x, y]))
    axs[1].plot(range(y.shape[0]), means_cup, color="red")

    plt.show()


def val_loop(dataloader, model, loss_fn, tag="Val"):
    size = len(dataloader.dataset)
    val_loss, accuracy, within, utilization = 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()

            means = pred[:, :, 0]
            variances = pred[:, :, 1]
            accuracy += torch.sum(torch.abs(means - y) / y).item()
            within += torch.sum(y <= means + LATITUDE_FACTOR * torch.sqrt(variances))
            utilization += torch.sum(y / (means + LATITUDE_FACTOR * torch.sqrt(variances)))

    val_loss /= (size * PERIOD)
    log_printf(TASK_ID, tag + " " + f"Error: \n Accuracy: {100 - (100 * accuracy / (size * PERIOD)):>0.3f}%, Avg loss: {val_loss:>8f}")
    log_printf(TASK_ID, f" Within the Power Generation: {(100 * within / (size * PERIOD)):>0.3f}%")
    log_printf(TASK_ID, f" Utilization Rate:  {(100 * utilization / (size * PERIOD)):>0.3f}%\n")

    return val_loss, 1 - accuracy / (size * PERIOD)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # check_gradient_norm(model)
        # clip
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} Avg loss: {loss / (X.shape[0] * PERIOD) :>7f}  [{current:>5d}/{size:>5d}]")


loss_function = customized_loss

def train_model():
    dataset = LD2011_2014_summary_by_day(length=LENGTH,
                                         csv_file=r"dataset/LD2011_2014.csv",
                                         )
    train, val, test = construct_dataloader(dataset, batch_size=BATCH_SIZE)
    print(len(dataset))

    energy_expectations, energy_variances = dataset.statistics()

    predictor = PeriodicalModel(input_size=PERIOD, hidden_size=HIDDEN_SIZE, num_layers=1, output_size=PERIOD,
                                batch_size=BATCH_SIZE, period=PERIOD,
                                means=energy_expectations)

    print(TASK_ID + ' model:', predictor)
    # loss_function = nn.MSELoss()
    adam = torch.optim.Adam(predictor.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)

    best_model = None
    min_val_loss = 50000000000
    tolerance = 0
    print('train_sum=', len(train))

    val_loop(val, predictor, loss_function)
    for epoch in range(EPOCH_STEP):
        print("========EPOCH {}========\n".format(epoch))
        train_loop(train, predictor, loss_function, adam)
        validation_loss, bias = val_loop(val, predictor, loss_function)
        tolerance += 1
        if validation_loss < min_val_loss:
            min_val_loss = validation_loss
            best_model = copy.deepcopy(predictor)
            tolerance = 0
        if tolerance > TOLERANCE:
            log_printf(TASK_ID, "Early stopped at epoch {}.\n".format(epoch))
            break

    best_model.lstm.flatten_parameters()
    performance_log(TASK_ID, "========Best Performance========\n", model=predictor)
    val_loop(val, best_model, loss_function)
    val_loop(test, best_model, loss_function, tag="Test")
    plot_forecasting_random_samples(best_model, test.dataset, LATITUDE_FACTOR, filename="Performance")


def test_model():
    predictor = PeriodicalModel(input_size=PERIOD, hidden_size=HIDDEN_SIZE, num_layers=1, output_size=PERIOD,
                                batch_size=BATCH_SIZE, period=PERIOD)
    predictor.load_state_dict(load_task_model(TASK_ID))
    predictor.eval()

    dataset = LD2011_2014_summary_by_day(length=LENGTH,
                                         csv_file=r"dataset/LD2011_2014.csv",
                                         )
    train, val, test = construct_dataloader(dataset, batch_size=BATCH_SIZE)
    with mute_log():
        val_loop(val, predictor, loss_function, tag="Val")
        val_loop(test, predictor, loss_function, tag="Test")
        plot_forecasting_random_samples(predictor, test.dataset, LATITUDE_FACTOR, filename="Performance")


if __name__ == "__main__":
    test_model()
    # train_model()