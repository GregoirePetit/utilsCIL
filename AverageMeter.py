import pickle
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PerfMonitor():
    """
    Stores the history of values to analyse them
    #TODO implement plot function with matplotlib in a separate script
    """
    def __init__(self) : 
        self.reset()

    def reset(self):
        self.history = []

    def stats(self) : 
        self.avg = np.mean(self.history)
        self.count = len(self.history)

    def update(self, val):
        self.history.append(val)

    def save_history(self, save_path=None):
        if save_path is not None : 
            with open(save_path, "wb") as file_path : 
                pickle.dump(self.history, file_path)

    def open_history(self, open_path=None) : 
        if open_path is not None : 
            with open(open_path, "rb") as file_path : 
                self.history = pickle.load(file_path)



