# Code partially copied from:
# https://github.com/ghosthamlet/deepproblog/blob/master/examples/NIPS/MNIST/single_digit/generate_data.py


import torchvision
import torch
import torchvision.transforms as transforms
import random
import pickle

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]) #, 0.5), (0.5, 0.5, 0.5))])
mnist_train_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True,transform=transform)
mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True,transform=transform)


def next_example(dataset,i):
    x, y = next(i),next(i)
    (x, c1), (y, c2) = dataset[x], dataset[y]

    s = c1 + c2
    label = [0.0] * 19
    label[s] = 1.0
    label = torch.tensor([label])

    return x, y, label


def gather_examples(dataset,filename, n):
    examples = list()
    i = list(range(len(dataset)))
    random.shuffle(i)
    i = iter(i)

    for _ in range(n):
        try:
            examples.append(next_example(dataset,i))
        except StopIteration:
            break

    with open(filename, 'wb') as f:
        pickle.dump(examples, f)


gather_examples(mnist_train_data, 'train_data', 3000)
gather_examples(mnist_test_data,'test_data', 5000)

