import torch
import torch.nn as nn
from torchvision.models import vgg16 

class VGG16(nn.Module):
    def __init__(self, dataset='cifar10', num_classes=10, pretrained = False):
        super(VGG16, self).__init__()
        assert dataset in ['cifar10', 'cifar100']

        data_mean = [0.5, 0.5, 0.5]
        data_std = [0.2, 0.2, 0.2]

        self.mean = nn.Parameter(torch.tensor(data_mean).unsqueeze(0).unsqueeze(2).unsqueeze(3),requires_grad=False)
        self.std = nn.Parameter(torch.tensor(data_std).unsqueeze(0).unsqueeze(2).unsqueeze(3),requires_grad=False)

        self.net = vgg16(pretrained=pretrained)    
        try:
            del self.net.avgpool
            self.net.avgpool = lambda x: x
        except:
            print('This version of torchvision does not have avgpooling in VGG16')
        # Change last linear layers
        lin1_inp_size = 512
        lin1_out_size = 512
        lin2_inp_size = 512
        lin2_out_size = 512
        lin3_inp_size = 512
        self.net.classifier._modules['0'] = nn.Linear(lin1_inp_size, lin1_out_size)
        self.net.classifier._modules['3'] = nn.Linear(lin2_inp_size, lin2_out_size)
        self.net.classifier._modules['6'] = nn.Linear(lin3_inp_size, num_classes)


    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x)