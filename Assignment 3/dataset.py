import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torch.nn.init import xavier_uniform_
from torchvision import transforms

# define your dataset class
class CIFAR10Dataset(Dataset):
  def __init__(self, X, y, train=False):
    self.X = torch.from_numpy(X.transpose([0, 3, 1, 2]))
    self.y = torch.from_numpy(y)
    self.train = train
    # self.transform = transforms.Compose([
    #                                      transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
    #                                      transforms.ColorJitter(hue=.05, saturation=.05),
    #                                      transforms.RandomHorizontalFlip(),
    #                                      transforms.RandomVerticalFlip()
    #                                      ])
    self.transform = transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # self.X.to(device)
    # self.y.to(device)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    # if self.train:
    #   return [self.transform(self.X[idx]), self.y[idx]]
    # else:
    #   return [self.X[idx], self.y[idx]]
    return [self.transform(self.X[idx]), self.y[idx]]
