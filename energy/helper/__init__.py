from collections import namedtuple
from contextlib import contextmanager

LOG_DIRECTORY = r"log"

TrainingProcess = namedtuple('TrainingProcess', ['gradient_norm', 'train_loss', 'val_loss'])
training_recoder = dict()

# 用于静音log
is_muted = False
@contextmanager
def mute_log_plot():
    global is_muted
    is_muted = True
    yield
    is_muted = False

