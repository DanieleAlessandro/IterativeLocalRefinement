from generate_knowledge import *
from evaluation import *
from models import *
import torchvision
import torchvision.transforms as transforms
from dataloader import *
import time

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True,transform=transform)

train_loader = dataloader('train_data', 128)
test_loader = dataloader('test_data', 5000)

#
# # ===== Model and learning settings =====
nn = MNIST_Net()
LRL = MNIST_ILRModel(nn, AND(knowledge))

optimizer = torch.optim.Adam(LRL.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()
print('Metrics before training:')
test_MNIST(nn, mnist_test_data)
print('Accuracy in sum task:' + str(test_sum(LRL, test_loader)))

epochs = 100
# Learning
for e in range(epochs):
    epoch_start = time.time()
    for i,(x,y,l) in enumerate(train_loader):
        optimizer.zero_grad()
        s_loss = loss(LRL(x, y)[0], torch.squeeze(l))
        s_loss.backward()
        optimizer.step()

    print(f'End of epoch {e}')
    print('Epoch time: ', time.time() - epoch_start)
    print('Accuracy in sum task: ' + str(test_sum(LRL, test_loader)))
    if e % 5 == 0:
        test_MNIST(nn, mnist_test_data)
