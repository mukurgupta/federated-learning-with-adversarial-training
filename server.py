import torch
import torch.nn as nn
import numpy as np
import os

import models
import user
from pgd import PGDAttack

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.autograd.set_detect_anomaly(True)

class Server:
    def __init__(self, users, batch_size, learning_rate, fading_rate, momentum, data_set, model):
        self.criterion = nn.NLLLoss().to(device)
        self.users = users
        self.learning_rate = learning_rate
        self.fading_rate = fading_rate
        self.momentum = momentum
        self.data_set = data_set
        if data_set == models.CIFAR10:
            self.net = models.Cifar10Net(model)
        else:
            raise NotImplementedError
        self.test_net = self.test_net.to(device)
        self.test_loader = torch.utils.data.DataLoader(self.test_net.dataset(False), batch_size=int(batch_size/2), shuffle=False)

        self.current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in self.test_net.parameters()])
        self.users_grads = np.empty((len(users), len(self.current_weights)), dtype=self.current_weights.dtype)
        self.velocity = np.zeros(self.current_weights.shape, self.users_grads.dtype)


    def save_model(self, epochs):
        print("\n\n",self.test_net)
        filename = "epoch_{}".format(epochs) + '.pth'
        fileloc = os.path.join('./', filename) 

        with open(fileloc, 'wb') as file:
            torch.save(self.test_net.state_dict(), file)
        print("\n", "model saved")
        return

    def load_model(self, epoch):
        filename = "epoch_{}".format(epoch) + '.pth'
        fileloc = os.path.join('./', filename)
        self.test_net.load_state_dict(torch.load(fileloc, map_location=torch.device(device)))
        self.current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in self.test_net.parameters()])
        return

    def calc_learning_rate(self, cur_epoch):
        if cur_epoch>=150:
            return self.learning_rate*0.1
        elif cur_epoch>=200:
            return self.learning_rate*0.1*0.1
        return self.learning_rate

    def train_client(self, cur_epoch):
        for usr in self.users:
            usr.step(self.current_weights, self.calc_learning_rate(cur_epoch))

    def collect_gradients(self):
        for idx, usr in enumerate(self.users):
            self.users_grads[idx, :] = usr.grads

    def fedAvg(self):
        current_grads = np.mean(self.users_grads, axis=0)
        self.current_weights = current_grads

    def test(self):
        user.row_into_parameters(self.current_weights, self.test_net.parameters())
        test_loss = 0
        correct = 0

        self.test_net.eval()
        with torch.no_grad():
            for data, target in self.test_loader:

                data, target = data.to(device), target.to(device)
                net_out = self.test_net(data)
                loss = self.criterion(net_out, target)
                test_loss += loss.data.item()
                pred = net_out.data.max(1)[1]
                correct += pred.eq(target.data).sum()

        test_loss /= len(self.test_loader.dataset)

        return test_loss, correct

    def adv_test(self, eps=0.3):
        self.eps = eps
        user.row_into_parameters(self.current_weights, self.test_net.parameters())
        test_loss = 0
        correct = 0

        test_loader = self.test_loader
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = self.mix_adversarial(data, target)
            net_out = self.test_net(data)
            loss = self.criterion(net_out, target)
            test_loss += loss.data.item()
            pred = net_out.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        test_loss /= len(self.test_loader.dataset)

        return test_loss, correct

   
    def mix_adversarial(self, x, y):
        #Vary the PGD attack parameters
        attacker = PGDAttack(self.test_net, 2/255, 10, 2/255) 
        x = attacker.perturb(x, y)
        return x

