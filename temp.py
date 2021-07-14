import math
import torch
import pickle
import itertools
import numpy as np
from tqdm import tqdm

from util.kernel import sanitise, increment_kernel, increment_kernel_improved, complexity, invert_bound
from util.data import get_data, normalize_data
import torch
import torch.nn as nn
import torch.nn.functional as F
# from util.trainer import SimpleNet

class SimpleNet(nn.Module):
    def __init__(self, depth, width):
        super(SimpleNet, self).__init__()

        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 1, bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)


## Check compositional kernel

# a = torch.ones(5,5) * 0.4
# b = increment_kernel(a)
# print(b) # should be 0.5441

# a = torch.zeros(5,5)
# b = increment_kernel(a)
# print(b) # should be 0.3183
# print()

# ## Check kernel complexity on identity matrix

# n = 100
# sigma = torch.eye(n)
# c = torch.randn(n).sign()
# print( complexity(sigma, c, 10000) ) # should be n*ln(2), 7n/10
# print()

# ## Check kernel complexity via non-cholesky approach

# n = 10
# rand = torch.randn(n, n)
# sigma = torch.mm(rand, rand.t()) + torch.eye(n)
# sigma = 0.5*(sigma + sigma.t())
# assert ( sigma == sigma.t() ).all()

# c = torch.randn(n).sign()

# det = torch.det(sigma)
# inv = torch.inverse(sigma)
# tr = torch.trace(inv)

# comp_1 = n/5.0 + det ** (1/n) * ( (0.5-1/math.pi)*tr + 1/math.pi*torch.dot(c, torch.matmul(inv, c)) )
# comp_1 = comp_1.item()

# num_samples = 10**6
# ide = torch.eye(n)
# estimate = 0
# print("running non-parallelised estimation")
# for _ in tqdm(range(num_samples)):
# 	z = torch.randn(n).abs()
# 	estimate += math.exp( -0.5 * torch.dot(c*z, torch.matmul(det**(1/n)*inv - ide,c*z)) )
# estimate /= num_samples

# comp_0 = math.log(2**n / estimate)

# print( ("Estimate", "Bound"))
# print( "non-cholesky:")
# print( (comp_0, comp_1) )
# print( "cholesky:")
# print( complexity(sigma, c, num_samples) )
# print()

## Check kernel compared to random networks

depth = 3
width = 1000
num_train_examples = 5
num_networks = 10**3
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

_, _, train_loader, _ = get_data( num_train_examples=num_train_examples,
								  num_test_examples=None,
                                  batch_size=num_train_examples,
	                              random_labels=False,
	                              binary_digits=False )

for data, target in train_loader:
    data, target = normalize_data(data, target)

out_matrix = np.zeros((num_train_examples, num_networks))

with torch.no_grad():
	print(f"Sampling {num_networks} random networks")
	for network_idx in tqdm(range(num_networks)):
		model = SimpleNet(depth, width)
		for p in model.parameters():
			p.data = torch.randn_like(p) / math.sqrt(p.shape[1])

		pred = model(data).squeeze()
		out_matrix[:, network_idx] = pred.numpy()

sample_mean = np.mean(out_matrix, axis=1)
sample_cov = np.cov(out_matrix)

print()
print("sample mean:\n", sample_mean)
print("sample cov:\n", sample_cov)

sigma_1 = sanitise(torch.mm(data, data.t()) / data.shape[1])
sigma_improved = sanitise(torch.mm(data, data.t()) / data.shape[1])
for _ in range(depth-1):
    sigma_1 = increment_kernel(sigma_1)
    sigma_improved = increment_kernel_improved(sigma_improved)

sigma_1 = sigma_1.numpy()
sigma_improved = sigma_improved.numpy()

diff_baseline = np.absolute(sigma_1 - sample_cov).max()
diff_improved = np.absolute(sigma_improved - sample_cov).max()

# print("analytical cov:\n", sigma)
print("max difference (baseline): ", diff_baseline)
print("max difference (improved): ", diff_improved)