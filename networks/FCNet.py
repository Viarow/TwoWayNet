import torch
from torch import nn 
import numpy as np


class BinaryClassifier(nn.Module):

    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 2)

    def forward(self, y):
        x = self.fc1(y)
        x = self.relu(x)
        xhat = self.fc2(x)
        return xhat


class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, y):
        xhat = self.linear(y)
        return xhat


class FlatFadingClassifier(nn.Module):

    def __init__(self):
        super(FlatFadingClassifier, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.relu = nn.ReLU()

    def forward(self, y):
        x = self.fc1(y)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        xhat = self.fc3(x)
        return xhat



class QAM4Classifier(nn.Module):

    def __init__(self):
        # num_classes = modulation order
        super(QAM4Classifier, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 4)

    def forward(self, y):
        x = self.fc1(y)
        x = self.relu(x)
        xhat = self.fc2(x)
        return xhat


class MIMO_BPSK_MLP(nn.Module):

    def __init__(self, mod, N):
        super(MIMO_BPSK_MLP, self).__init__()
        num_classes = int(np.power(mod, N))
        self.fc1 = nn.Linear(N, N)
        self.fc2 = nn.Linear(N, num_classes)
        self.fc3 = nn.Linear(num_classes, num_classes)
        self.relu = nn.ReLU()

    def forward(self, y):
        x = self.fc1(y)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        xhat = self.fc3(x)
        return xhat