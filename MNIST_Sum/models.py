import torch.nn as nn
import os
from logical_layers import *

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

# Architecture copied from:
# https://github.com/ghosthamlet/deepproblog/blob/master/examples/NIPS/MNIST/mnist.py
class MNIST_Net(nn.Module):
    def __init__(self, N=10):
        super(MNIST_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  6, 5),
            nn.MaxPool2d(2, 2), # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2), # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier =  nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x


class MNIST_ILRModel(nn.Module):
    def __init__(self, nn, formula):
        super(MNIST_ILRModel, self).__init__()
        self.nn = nn
        self.layer = LRL(formula, 1, method='max', residuum=True)

    def forward(self, x, y):
        x = self.nn(x)
        y = self.nn(y)
        consequents = torch.zeros([x.shape[0], 19])
        return self.layer(torch.concat([x,y,consequents], dim=1), 1.0)[:, 20:]