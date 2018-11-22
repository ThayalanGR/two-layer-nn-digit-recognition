
# Homecoming (eYRC-2018): Task 1A
# Build a Fully Connected 2-Layer Neural Network to Classify Digits

# NOTE: You can only use Tensor API of PyTorch

from nnet import model

# TODO: import torch and torchvision libraries
# We will use torchvision's transforms and datasets
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# TODO: Defining torchvision transforms for preprocessing

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# TODO: Using torchvision datasets to load MNIST
root = './data'
if not os.path.exists(root):
    os.mkdir(root)


# TODO: Use torch.utils.data.DataLoader to create loaders for train and test
# NOTE: Use training batch size = 4 in train data loader.


train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=True)

# NOTE: Don't change these settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28  # Input size
N_h1 = 256  # Hidden Layer 1 size
N_h2 = 256  # Hidden Layer 2 size
N_out = 10  # Output size
# Learning rate
lr = 0.001


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

# TODO: Define number of epochs
N_epoch = 5  # Or keep it as is 


dataiter = iter(train_loader)
images, labels = dataiter.next()
# data = enumerate(train_loader)
# idx, (inputs, labels) = next(data)
# print(idx)
# print(inputs, labels)

# TODO: Training and Validation Loop
# >>> for n epochs
# >>> for all mini batches
# >>> net.train(...)
# at the end of each training epoch
# >>> net.eval(...)
for i in range(0, N_epoch):
    dataiter = iter(train_loader)
    inputs, labels = dataiter.next()
    # print(labels)
    net.train(inputs, labels, lr, debug=False)
    net.eval(inputs, labels, debug=False)


# TODO: End of Training
# make predictions on randomly selected test examples
# >>> net.predict(...)

dataiter = iter(test_loader)
inputs, labels = dataiter.next()
# predictions takes inputs and throws Output
net.predict(inputs)
