import pickle
import numpy as np
import torch


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

    def save(self, save_path=None):
        #print("DEBUG saving perf monitor")
        #print(self.history)
        #print(self.history[0])
        #print(type(self.history[0]))
        if save_path is not None : 
            with open(save_path, "wb") as file_path : 
                pickle.dump(self.history, file_path)

    def open(self, open_path=None) : 
        if open_path is not None : 
            with open(open_path, "rb") as file_path : 
                self.history = pickle.load(file_path)


class ConfusionMatrix(object):
    """
    Important note : Confusion matrix for 1 category level in the class hirarchy i.e.
    you must compute separately a confusion matrix for fine classes and for coarse classes.
    """

    def __init__(self, n_classes, multilabel=False):
        self.matrix = torch.zeros(n_classes, n_classes)
        self.n_classes = n_classes
        self.multilabel = multilabel

    def update(self, output, target):
        """
        Within 1 epoch, count across batches of examples
        Within 1 incremental step, count across batches of examples
        """
        #print("output : ", output.size())
        #print("target : ", target.size())
        batch_size = target.size(0)
        
        if self.multilabel == True : 
            assert(output.size() == target.size())
            # get the predictions using the decision threshold
            multilabel_pred = (output > 0.5)*1.0
            #print(multilabel_pred)
            for i in range(batch_size) : 
                target_index, pred_vect = torch.argmax(target[i]), multilabel_pred[i]
                self.matrix[target_index]+=pred_vect 
                #print("\nSample ", i)
                #print("target_index ", target_index)
                #print("pred_vect ", pred_vect)
                #print(self.matrix)           
        
        else : # classic 1 label
            for i in range(batch_size) : 
                pred = output 
                target_index, pred_index = target[i], torch.argmax(pred[i])
                self.matrix[target_index][pred_index]+=1
                #print("\nSample ", i)
                #print("target_index ", target_index)  
                #print("pred_index", pred_index)
                #print(self.matrix)
        
        return None

    def save(self, save_path=None):
        if save_path is not None : 
            self.matrix = self.matrix.detach().cpu().numpy()
            with open(save_path, "wb") as file_path : 
                pickle.dump(self.matrix, file_path)
        return None






