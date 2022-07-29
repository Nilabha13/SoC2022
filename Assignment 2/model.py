import numpy as np
from matplotlib import pyplot as plt
import torch
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform_

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.lin1 = nn.Linear(784, 40)
    self.relu1 = nn.ReLU()
    self.lin2 = nn.Linear(40, 30)
    self.relu2 = nn.ReLU()
    self.lin3 = nn.Linear(30, 10)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x):
    x = self.lin1(x)
    x = self.relu1(x)
    x = self.lin2(x)
    x = self.relu2(x)
    x = self.lin3(x)
    x = self.softmax(x)

    return x
