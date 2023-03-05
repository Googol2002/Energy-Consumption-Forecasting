import torch.multiprocessing as mp


def process_runner(train, dataset, pipe_in):

    def runner(cuda_unit: str, remaining_cuda: mp.Semaphore, pipe_out: mp.SimpleQueue):
        train()

    return runner


class MultiTaskTrainer:

    def __init__(self, train, cuda_list, folds):
        self.train = train
        self.cuda_list = cuda_list
        self.folds = folds

    def dispatch(self):
        remaining_cuda = mp.Semaphore(len(self.cuda_list))
        processes = []
        for cuda_unit in self.cuda_list:
            pipe = mp.SimpleQueue()
            p = mp.Process(target=process_runner(self.train, pipe),
                           args=(cuda_unit, remaining_cuda, pipe))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()