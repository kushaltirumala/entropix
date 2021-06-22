import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from util.data import get_data, normalize_data
from util.trainer import SimpleNet, ResidualNet, JeremySimpleNet, ResidualNetVariancePreserving, ResidualNetVariancePreservingV2



seed = 100

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

device = torch.device("cuda:0")

num_networks = 5000
width = 1000
batch_size = 10
shuffle=True
num_workers = 1
binary_mode = True


def main():

    # depths = [2, 3, 5, 10, 20]
    depths = [2, 5, 10, 30]
    # depth = 2


    # alpha_vals = np.linspace(0, 10, 20)
    alpha_vals = np.linspace(0, 1, 10)
    # alpha = 0.1

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


    train_data_x, train_data_y = next(iter(training_loader))

    if binary_mode:
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

    empirical_heatmap = np.zeros((len(depths), len(alpha_vals)))

    for i, depth in enumerate(tqdm(depths)):
        for j, alpha in enumerate(alpha_vals):
            num_perfect_networks = 0

            with torch.no_grad():
                # print(f"Sampling {num_networks} random networks")
                for network_idx in range(num_networks):
                    model = ResidualNetVariancePreservingV2(depth, width, alpha)
                    model = model.to(device)
                    for p in model.parameters():
                        p.data = torch.randn_like(p) / math.sqrt(p.shape[1])

                    pred = model(data).squeeze()

                    if binary_mode:
                        predicted_labels = torch.sign(pred)
                    else:
                        predicted_labels = torch.argmax(pred, axis=1)


                    num_match_elements = torch.sum(predicted_labels == labels).item()
                    if num_match_elements == labels.shape[0]:
                        # this means the entire data was predicted correctly:
                        num_perfect_networks += 1

            print(empirical_heatmap)

            empirical_heatmap[i][j] = num_perfect_networks


    print(empirical_heatmap)
    np.save(open(f"v3_kernel/empirical_heatmap_results_seed_{seed}.npy", "wb"), empirical_heatmap)

            


if __name__ == "__main__":
    main()
