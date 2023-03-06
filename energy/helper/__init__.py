from collections import namedtuple
from contextlib import contextmanager

from datetime import datetime, timedelta, timezone

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


def current_time_tag():
    # 获取当前上海时间
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)
    shanghai_time = utc_time.astimezone(SHA_TZ)
    # 转换为其他日期格式，如："%Y-%m-%d %H:%M:%S"
    return shanghai_time.strftime("%Y-%m-%d %H-%M-%S")
