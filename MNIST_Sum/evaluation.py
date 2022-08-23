from torch.autograd import Variable
import numpy as np
import torch


# Function copied from:
# https://github.com/ghosthamlet/deepproblog/blob/master/examples/NIPS/MNIST/mnist.py
def test_MNIST(model, dataset, max_digit=10):
    confusion = np.zeros((max_digit,max_digit),dtype=np.uint32) # First index actual, second index predicted
    N = 0
    for d,l in dataset:
        if l < max_digit:
            N += 1
            d = Variable(d.unsqueeze(0))
            outputs = model(d)
            _, out = torch.max(outputs.data, 1)
            c = int(out.squeeze())
            confusion[l,c] += 1
    print(confusion)
    F1 = 0
    for nr in range(max_digit):
        TP = confusion[nr,nr]
        FP = sum(confusion[:,nr])-TP
        FN = sum(confusion[nr,:])-TP
        F1 += 2*TP/(2*TP+FP+FN)*(FN+TP)/N
    print('F1: ',F1)
    return [('F1',F1)]


def test_sum(model, dataloader):
    x, y, s = next(iter(dataloader))

    _, prediction = torch.max(model(x, y)[1], 1)
    _, label = torch.max(torch.squeeze(s), 1)

    return torch.sum(label == prediction) / label.shape[0]
