from generate_knowledge import *
from evaluation import *
from models import *
import torchvision
import torchvision.transforms as transforms
from dataloader import *
import time

# Settings
epochs = 30
n_runs = 10
learning_rate = 0.01
batch_size = 128

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True,transform=transform)

train_loader = dataloader('train_data', batch_size)
test_loader = dataloader('test_data', 5000)

results_accuracy = []
results_f1 = []
epoch_times = []
for run in range(n_runs):
    print(f'Starting run {run}')

    # # ===== Model and learning settings =====
    nn = MNIST_Net()
    LRL = MNIST_ILRModel(nn, AND(knowledge))

    optimizer = torch.optim.Adam(LRL.parameters(), lr=learning_rate)
    loss = torch.nn.BCELoss()
    print('Metrics before training:')
    test_MNIST(nn, mnist_test_data)
    print('Accuracy in sum task:' + str(test_sum(LRL, test_loader)))

    # Learning
    for e in range(epochs):
        epoch_start = time.time()
        for i,(x,y,l) in enumerate(train_loader):
            optimizer.zero_grad()

            s_loss = loss(LRL(x, y), torch.squeeze(l))
            s_loss.backward()
            optimizer.step()

        print(f'End of epoch {e}')
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print('Epoch time: ', epoch_time)
        test_accuracy = test_sum(LRL, test_loader)
        print('Accuracy in sum task: ' + str(test_accuracy))
        if e % 15 == 0 and e != 0:
            test_MNIST(nn, mnist_test_data)

    f1 = test_MNIST(nn, mnist_test_data)
    results_accuracy.append(test_accuracy.cpu().detach().numpy())
    results_f1.append(f1)

print(f'Average of accuracy in the MNIST sum after {n_runs} runs: {np.mean(results_accuracy)}, std: {np.std(results_accuracy)}')
print(f'Average of f1 score for the MNIST digits after {n_runs} runs: {np.mean(results_f1)}, std: {np.std(results_f1)}')
print(f'Average seconds per epoch: {np.mean(epoch_times)}')

print('All accuracies')
print([float(a) for a in results_accuracy])
print('All f1 scores')
print(results_f1)
