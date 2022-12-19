import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import itertools
import pandas as pd

class L3FeaturesDataset(Dataset):

    def __init__(self, path, range_classes=None):
        self.y = []
        self.X = []
        doss = os.listdir(path)
        if range_classes:
            doss = [f'{d}_decomposed' for d in range_classes]
        doss = np.sort(doss)
        for root_path in doss:
            if '_' in root_path:
                for file in os.listdir(os.path.join(path, root_path)):
                    self.y.append(int(root_path.split('_')[0]))
                    self.X.append(os.path.join(path, root_path, file))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        label = self.y[index]
        with open(self.X[index], "r") as f:
            image = np.array([float(x) for x in f.read().split()]).reshape(256,14,14).astype(np.float32)
        return image, label

class L3FeaturesDatasetFeatureMaps(Dataset):

    def __init__(self, path, range_classes=None):
        self.y = []
        self.X = []
        doss = os.listdir(path)
        if range_classes:
            doss = [f'{d}_decomposed' for d in range_classes]
        doss = np.sort(doss)
        for root_path in doss:
            if '_' in root_path:
                for file in os.listdir(os.path.join(path, root_path)):
                    self.y.append(int(root_path.split('_')[0]))
                    self.X.append(os.path.join(path, root_path, file))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        label = self.y[index]
        with open(self.X[index], "r") as f:
            image = np.array([float(x) for x in f.read().split()]).reshape(14,256*14).astype(np.float32)
        return image, label

class L4FeaturesDataset(Dataset):

    def __init__(self, path, range_classes=None):
        self.y = []
        self.X = []
        doss = os.listdir(path)
        if range_classes:
            doss = [f'{d}_decomposed' for d in range_classes]
        doss = np.sort(doss)
        for root_path in doss:
            if '_' in root_path:
                for file in os.listdir(os.path.join(path, root_path)):
                    self.y.append(int(root_path.split('_')[0]))
                    self.X.append(os.path.join(path, root_path, file))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        label = self.y[index]
        with open(self.X[index], "r") as f:
            image = np.array([float(x) for x in f.read().split()]).astype(np.float32)
        return image, label