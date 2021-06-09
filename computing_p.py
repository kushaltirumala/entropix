import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

batch_size = 10
shuffle=True
num_workers = 1
depth = 2

seed = 1
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


def sanitise(sigma):
    return torch.clamp(sigma, -1, 1)


class PBoundNetwork(nn.Module):
    def __init__(self):
        super(PBoundNetwork, self).__init__()
        # self.layers = nn.ModuleList([HFunction() for _ in range(depth-1)])

    def forward(self, x, c, alpha):

        sigma_1 = sanitise((torch.mm(x, x.t())))

        for _ in range(depth - 1):
            new_sigma_1 = torch.sqrt(1-sigma_1**2)
            new_sigma_added_1 = new_sigma_1 + sigma_1*(math.pi - torch.acos(sigma_1))
            final_sigma_1 = new_sigma_added_1/math.pi
            pre_clamped_sig_1 = (1.0/(1.0 + alpha**2)) * (sigma_1 + (alpha**2) * (final_sigma_1))
            sigma_1 = torch.clamp(pre_clamped_sig_1, -0.95, 0.95)

        # fill diagonals with 1's again
        n = sigma_1.shape[0]
        sigma_1.fill_diagonal_(1)
        id = torch.eye(n)

        u = torch.cholesky(sigma_1)

        inv = torch.cholesky_inverse(u)

        nth_root_det = u.diag().pow(2/n).prod()
        inv_trace = inv.diag().sum()
        inv_proj = torch.dot(c, torch.mv(inv, c))

        formula_1 = nth_root_det * ( (0.5 - 1/math.pi)*inv_trace + inv_proj/math.pi )

        return formula_1

def main():

    trainset = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    testset = datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))


    training_loader = DataLoader(trainset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,drop_last=False)
    test_loader = DataLoader(testset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,drop_last=False)


    train_data_x, train_data_y = next(iter(training_loader))
    test_data_x, test_data_y = next(iter(test_loader))
    train_data = train_data_x
    train_data_label = train_data_y
    test_data = test_data_x
    test_data_label = test_data_y

    data = train_data
    labels = train_data_label


    data = torch.flatten(data, start_dim = 2, end_dim = 3).squeeze()

    model = PBoundNetwork()

    torch.autograd.set_detect_anomaly(True)

    alpha = Variable(torch.Tensor([0.5]), requires_grad=True)

    # alpha = 0.1
    norms = torch.norm(data, dim=1)
    normed_data_final = torch.div(data.T, norms).T

    labels = labels.type(torch.FloatTensor)
    outputs = model(normed_data_final, labels, alpha)

    outputs.backward()

    print(alpha.grad)




if __name__ == "__main__":
    main()
