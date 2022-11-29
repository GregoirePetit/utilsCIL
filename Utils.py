import sys, numpy, random, re, os
# from MyStaticSamplers import *
import numpy as np
from tqdm import tqdm
import torch

# TODO : confusion matrix, classification report, version multilabel coarse et fine precision et recall

class DataUtils():
    def __init__(self):
        return

    def accuracy_multilabel(self, output, target, topk=(1,), threshold = 0.5, split_index=None):
        """Computes the accuracy for an exact match of both fine and coarse classes based on the multilabel prediction"""
        batch_size = target.size(0)
        #print("output : ", output.size())
        #print("target : ", target.size())
        assert(output.size() == target.size())
        # get the predictions using the decision threshold
        multilabel_pred = (output > 0.5)*1.0
        #print("DEBUG 1 : ", torch.sum(multilabel_pred))
        # get the term to term matches with the target labels + condition target true
        match = (multilabel_pred == target)*target # size batch_size * num_total_classes
        #print("DEBUG 2 : ", torch.sum(match))
        # split the tensor into fine and coarse component
        match_fine = match[:, :split_index]
        match_coarse = match[:, split_index:]
        #print(match_fine.size(), match_coarse.size())
        # compute the number of exact matches per category (only 1 true label for each category)
        n_match_fine = torch.sum(match_fine, dim=1)==1.0
        n_match_coarse = torch.sum(match_coarse, dim=1)==1.0
        #print(n_match_fine.size(), n_match_coarse.size())
        #print(torch.sum(match_fine, dim=1))
        #print(n_match_fine)
        #print(torch.sum(match_coarse, dim=1))
        #print(n_match_coarse)
        # sum the number of exact matches and devide it by the number of examples *100
        acc_fine = torch.sum(n_match_fine)*100.0/batch_size
        acc_coarse = torch.sum(n_match_coarse)*100.0/batch_size
        #print(acc_fine, acc_coarse)
        return acc_fine, acc_coarse

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def get_dataset_mean_std(self, normalization_dataset_name, datasets_mean_std_file_path):
        datasets_mean_std_file = open(datasets_mean_std_file_path, 'r').readlines()
        for line in datasets_mean_std_file:
            line = line.strip().split(':')
            dataset_name, dataset_stat  = line[0], line[1]
            if dataset_name == normalization_dataset_name:
                dataset_stat = dataset_stat.split(';')
                dataset_mean = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[0])]
                dataset_std = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[1])]
                return dataset_mean, dataset_std
        print('Invalid normalization dataset name')
        sys.exit(-1)


    def from_str_to_list(self, string, type):
        list = []
        params = string.split(',')
        for p in params:
            if type == 'int':
                list.append(int(p.strip()))
            elif type == 'float':
                list.append(float(p.strip()))
            elif type == 'str':
                list.append(str(p.strip()))
        return list







