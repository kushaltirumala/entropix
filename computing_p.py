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

    def forward(self, x, c, alpha):

        n = x.shape[0]

        sigma_1 = sanitise((torch.mm(x, x.t()) / x.shape[1]))


        for _ in range(depth-1):
        	new_sigma = torch.zeros((n, n))

        	sigma = sigma_1


        	for i in range(n):
        		for j in range(n):
        			k_x_x = sigma[i][i]
        			k_tilde_x_tilde_x = sigma[j][j]
        			cross_product_term = torch.sqrt(k_x_x * k_tilde_x_tilde_x)

        			mean_k_term = sigma[i][j] / cross_product_term
        			# mean_k_term = sigma[i][j]


        			new_mean_k_term = torch.sqrt(1-mean_k_term**2)
        			new_mean_k_term += mean_k_term*(math.pi - torch.acos(mean_k_term))
        			new_mean_k_term /= math.pi

        			# h_mean_k_term = (1.0/math.pi) * (torch.sqrt(1 - mean_k_term**2) + mean_k_term * (math.pi - torch.acos(mean_k_term)))

        			t_sigma_term = cross_product_term * new_mean_k_term

        			# assert(cross_product_term == 1)
        			# import pdb; pdb.set_trace()

        			new_sigma[i][j] = (1.0/(1.0 + alpha**2)) * (sigma[i][j] + (alpha**2) * t_sigma_term)
        			# new_sigma[i][j] = new_mean_k_term

        	new_sigma = torch.clamp(new_sigma, -1, 1)
        	sigma_1 = new_sigma


        n = sigma_1.shape[0]
        id = torch.eye(n)

        assert ( sigma_1 == sigma_1.t() ).all()
        det_sigma = torch.det(sigma_1)
        # if det_sigma == 0.0:
        #     # add a little noise to covariance to make it nonzero
        #     sigma[0][0] += 0.0001
        u = torch.cholesky(sigma_1)
        inv = torch.cholesky_inverse(u)
        assert (torch.mm(sigma_1, inv) - id).abs().max() < 1e-03

        nth_root_det = u.diag().pow(2/n).prod()
        inv_trace = inv.diag().sum()
        inv_proj = torch.dot(c, torch.mv(inv, c))

        formula_1 = n/5.0 + nth_root_det * ( (0.5 - 1/math.pi)*inv_trace + inv_proj/math.pi )
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

    alpha = Variable(torch.Tensor([1.0]), requires_grad=True)

    # alpha = 0.1

    labels = labels.type(torch.FloatTensor)
    outputs = model(data, labels, alpha)



    outputs.backward()

    print(alpha.grad)




if __name__ == "__main__":
    main()
