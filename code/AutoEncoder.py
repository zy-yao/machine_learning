import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

#Hyperparameters 
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = True
N_TEST_IMG = 5

#Mnist digits data
train_data = torchvision.datasets.MNIST(root = './mnist',
                                      train = True,
                                      transform = torchvision.transforms.ToTensor(),
                                      download = DOWNLOAD_MNIST)



