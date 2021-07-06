import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from util.data import get_data, normalize_data

from absl import flags
from absl import app


FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'seed for experiment run')




def sanitise(sigma):
    return torch.clamp(sigma, -1, 1)


class PBoundNetwork(nn.Module):
    def __init__(self):
        super(PBoundNetwork, self).__init__()
        # self.layers = nn.ModuleList([HFunction() for _ in range(depth-1)])

    def forward(self, x, c, alpha, depth):

        sigma_1 = sanitise((torch.mm(x, x.t())))

        for _ in range(depth - 1):
            new_sigma_1 = torch.sqrt(1-sigma_1**2)
            new_sigma_added_1 = new_sigma_1 + sigma_1*(math.pi - torch.acos(sigma_1))
            final_sigma_1 = new_sigma_added_1/math.pi
            
            pre_clamped_sig_1 = ((1 - alpha)*sigma_1 + (alpha) * (final_sigma_1))
            # pre_clamped_sig_1 = (1.0/(1.0 + alpha**2)) * (sigma_1 + (alpha**2) * (final_sigma_1))
            sigma_1 = torch.clamp(pre_clamped_sig_1, -0.95, 0.95)

        # fill diagonals with 1's again
        n = sigma_1.shape[0]
        sigma_1.fill_diagonal_(1)
        id = torch.eye(n)

        try:
            u = torch.cholesky(sigma_1)
        except RuntimeError as e:
            sigma_new = sigma_1 + 1.0*torch.eye(n)
            u = torch.cholesky(sigma_new)

        inv = torch.cholesky_inverse(u)

        nth_root_det = u.diag().pow(2/n).prod()
        inv_trace = inv.diag().sum()
        inv_proj = torch.dot(c, torch.mv(inv, c))

        formula_1 = nth_root_det * ( (0.5 - 1/math.pi)*inv_trace + inv_proj/math.pi )

        return formula_1

def main(argv):

    seed = FLAGS.seed

    batch_size = 2
    shuffle=True
    num_workers = 1

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    device = torch.device("cpu")

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


    lowest_c1_alpha_val = []

    training_loader = DataLoader(trainset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,drop_last=False)


    train_data_x, train_data_y = next(iter(training_loader))
    train_data_x = train_data_x.to(device)
    train_data_y = train_data_y.to(device)

    for i, label in enumerate(train_data_y):
        if label % 2 == 0:
            train_data_y[i] = -1
        else:
            train_data_y[i] = 1
    train_data = train_data_x
    train_data_label = train_data_y

    data = train_data
    labels = train_data_label

    data = torch.flatten(data, start_dim = 2, end_dim = 3).squeeze()
    norms = torch.norm(data, dim=1)
    normed_data_final = torch.div(data.T, norms).T
    data = normed_data_final
    labels = labels.type(torch.FloatTensor)

    data = data.to(device)
    labels = labels.to(device)

    model = PBoundNetwork()
    model = model.to(device)

    # torch.autograd.set_detect_anomaly(True)

    depth_vals = [2, 3, 5, 10]
    for depth in tqdm(depth_vals):
        # results_arr = []
        outputs_arr = []
        alpha_vals = np.linspace(0, 1, 100)
        for alpha_val in alpha_vals:
            alpha = Variable(torch.Tensor([alpha_val]), requires_grad=True)
            alpha = alpha.to(device)
            if alpha.grad is not None:
                alpha.grad.data.zero_()
            outputs = model(data, labels, alpha, depth)

            outputs.backward()

            # print("Alpha val: " + str(alpha_val) + " output: " + str(outputs.item()) + "grad: " + str(alpha.grad.item()))
            # results_arr.append(alpha.grad.item())
            outputs_arr.append(outputs.item())



        # results_arr = np.array(results_arr)
        outputs_arr = np.array(outputs_arr)

        # plt.plot(alpha_vals, results_arr)
        # plt.title("Alpha vals vs grad(C_1)")
        # plt.show()

        # plt.plot(alpha_vals, outputs_arr)
        # plt.title("Alpha vals vs C_1")
        # plt.show()

            # print("lowest C_1 bound happens at ")
        index_min = np.argmin(outputs_arr)
        alpha_val_index_min = alpha_vals[index_min]
        print("lowest alpha val: " + str(alpha_val_index_min))

        # np.save(open("v3_kernel/outputs_arr_new_formulation.npy", "wb"), outputs_arr)
        # np.save(open("v3_kernel/results_arr_30000.npy", "wb"), results_arr)
        lowest_c1_alpha_val.append(alpha_val_index_min)


    lowest_c1_alpha_val = np.array(lowest_c1_alpha_val)
    # plt.plot(depth_vals, lowest_c1_alpha_val)
    # plt.title("Depth vs optimal alpha value (fixed batch size = 10) MNIST, hard binary digits")
    # plt.show()

    np.save(open(f"full_run_v4_kernel_theoretical/lowest_c1_alpha_val_varying_depths_batch_10_seed_{seed}_bin_labels.npy", "wb"), lowest_c1_alpha_val)

if __name__ == "__main__":
    app.run(main)
