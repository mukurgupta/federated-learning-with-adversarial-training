import functools
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import models

from pgd import PGDAttack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def flatten_params(params):
    return np.concatenate([i.data.cpu().numpy().flatten() for i in params])

def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x,y:x*y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


class User:
    def __init__(self, user_id, batch_size, users_count, momentum, data_set, model, adversarial_training=0):
        self.user_id = user_id
        self.criterion = nn.NLLLoss().to(device)
        self.learning_rate = None
        self.grads = None
        self.data_set = data_set
        self.adversarial_training = adversarial_training
        self.momentum = momentum
        if data_set == models.CIFAR10:
            self.net = models.Cifar10Net(model)
        else:
            raise NotImplementedError
        self.net = self.net.to(device)
        self.original_params = None
        dataset = self.net.dataset(True)
        sampler = None
        if users_count > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=users_count, rank=user_id)
        self.train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler,batch_size=batch_size, shuffle=sampler is None)

    def train(self, data, target): 
        net_out = self.net(data)
        loss = self.criterion(net_out, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, current_params, learning_rate):
        row_into_parameters(current_params, self.net.parameters())
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=self.momentum, weight_decay=5e-4)

        for data, target in (self.train_loader):
            data, target = data.to(device), target.to(device)
            if self.adversarial_training:
                data = self.mix_adversarial(data, target)
            self.train(data, target)
        self.grads = np.concatenate([param.data.cpu().numpy().flatten() for param in self.net.parameters()])


    def mix_adversarial(self, x, y):
        #Replacing the 50% of the examples by Adversarial Examples
        rand_perm = np.random.permutation(x.size(0))
        rand_perm = rand_perm[:rand_perm.size//2]
        x_adv, y_adv = x[rand_perm,:], y[rand_perm]
        
        #Vary the PGD attack parameters
        attacker = PGDAttack(self.net, 2/255, 10, 2/255)
        x_adv = attacker.perturb(x_adv, y_adv)
        x[rand_perm,:] = x_adv
        return x