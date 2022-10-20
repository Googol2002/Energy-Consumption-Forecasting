import argparse

import torch

def _find_empty_device():
    global device
    for i in range(torch.cuda.device_count()):
        if torch.cuda.memory_allocated(device=0) < 1024*1024*256:
            device = "cuda:{}".format(i)
            return
    raise EnvironmentError("No Available Cuda kernel!")


parser = argparse.ArgumentParser(description='The Oracle是一个电力负载预测模型.')
parser.add_argument('--gpu', type=str, nargs=1,
                    help='设备名', metavar="cuda:0")
args = parser.parse_args()

device = "cuda:0"

if args.gpu is not None:
    device = torch.device(args.gpu[0] if torch.cuda.is_available() else "cpu")
else:
    _find_empty_device()



