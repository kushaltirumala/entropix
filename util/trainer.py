import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.nero import Nero
from util.data import normalize_data

# from tqdm import tqdm
tqdm = lambda x: x


class SimpleNet(nn.Module):
    def __init__(self, depth, width, alpha, residual=False):
        super(SimpleNet, self).__init__()

        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 1, bias=False)
        self.alpha = alpha
        self.residual = residual

    def forward(self, x):

        # if self.residual:
        #     x = x + self.alpha * F.relu(self.initial(x)) * math.sqrt(2)
        # else:
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)

        for layer in self.layers:
            if self.residual:
                x = x + self.alpha * F.relu(layer(x)) * math.sqrt(2)
            else:
                x = F.relu(layer(x)) * math.sqrt(2)

        # if self.residual:
        #     import pdb; pdb.set_trace()
        #     x = x + self.alpha * self.final(x)
        # else:
        x = self.final(x)

        return x


class ResidualNet(nn.Module):
    def __init__(self, depth, width, alpha):
        super(ResidualNet, self).__init__()

        self.width = width
        self.depth = depth
        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 1, bias=False)
        self.alpha = alpha

    def forward(self, x):

        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)

        for layer in self.layers:
            x = math.sqrt(1 - self.alpha**2)*x + self.alpha * F.relu(layer(x)) * math.sqrt(2)

        x = self.final(x)

        return x

class JeremySimpleNet(nn.Module):
    def __init__(self, depth, width):
        super(JeremySimpleNet, self).__init__()

        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 1, bias=False)
        self.width = width
        self.depth = depth

    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)


# def analyze_model(model):
#
#
#     for p in model.named_parameters():
#
#         true_param = p[1]
#         if true_param.requires_grad is None:
#             continue
#
#
#         import pdb; pdb.set_trace()
#
#         print("here")

def train_network(train_loader, test_loader, depth, width, init_lr, decay, cuda, alpha, break_on_fit=True):


    model = SimpleNet(depth, width, alpha, residual=True)
    if cuda:
        model = model.cuda()
    optim = Nero(model.parameters(), lr=init_lr)
    lr_lambda = lambda x: decay**x
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    train_acc_list = []
    train_acc = 0

    # import pdb; pdb.set_trace()

    for epoch in tqdm(range(100)):
        model.train()

        for data, target in train_loader:
            if cuda:
                data, target = (data.cuda(), target.cuda())
            data, target = normalize_data(data, target)

            y_pred = model(data).squeeze()
            loss = (y_pred - target).norm()

            model.zero_grad()
            loss.backward()
            optim.step()

            # analyze_model(model)


            # print("haha")

        lr_scheduler.step()

        model.eval()
        correct = 0
        total = 0

        for data, target in train_loader:
            if cuda:
                data, target = (data.cuda(), target.cuda())
            data, target = normalize_data(data, target)

            y_pred = model(data).squeeze()
            correct += (target.float() == y_pred.sign()).sum().item()
            total += target.shape[0]

        train_acc = correct/total
        train_acc_list.append(train_acc)

        if break_on_fit and train_acc == 1.0: break

    model.eval()
    correct = 0
    total = 0

    for data, target in test_loader:
        if cuda:
            data, target = (data.cuda(), target.cuda())
        data, target = normalize_data(data, target)

        y_pred = model(data).squeeze()
        correct += (target.float() == y_pred.sign()).sum().item()
        total += target.shape[0]

    test_acc = correct/total

    return train_acc_list, test_acc, model
