import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from vgg16.vgg16 import VGG16
from alexnet.alexnet import AlexNet

MNIST = 'MNIST'
CIFAR10 = 'CIFAR10'
CIFAR100 = 'CIFAR100'


class Cifar10Net(nn.Module):
    def __init__(self, model):
        super(Cifar10Net, self).__init__()
        if model == "VGG16":
            self.model = VGG16('cifar10',num_classes=10)
        elif model == "AlexNet":
            self.model = AlexNet('cifar10',num_classes=10)
        else:
            raise NotImplementedError

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

    @staticmethod
    def dataset(is_train, transform=None):
        if is_train:
            t = [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]
        else:
            t = [transforms.ToTensor()]

        if transform:
            t.append(transform)
        return datasets.CIFAR10(root='./cifar10_data', download=True, train=is_train, transform=transforms.Compose(t))