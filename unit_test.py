import math
import torch
import pickle
import itertools
import numpy as np
from tqdm import tqdm
import random

from util.kernel import sanitise, increment_kernel, complexity, invert_bound, increment_kernel_resnet_my_derivation
from util.data import get_data, normalize_data
from util.trainer import SimpleNet, ResidualNet, JeremySimpleNet

## Check compositional kernel
#
# a = torch.ones(5,5) * 0.4
# b = increment_kernel(a)
# print(b) # should be 0.5441
#
# a = torch.zeros(5,5)
# b = increment_kernel(a)
# print(b) # should be 0.3183
# print()
#
# ## Check kernel complexity on identity matrix
#
# n = 100
# sigma = torch.eye(n)
# c = torch.randn(n).sign()
# print( complexity(sigma, c, 10000) ) # should be n*ln(2), 7n/10
# print()
#
# ## Check kernel complexity via non-cholesky approach
#
# n = 10
# rand = torch.randn(n, n)
# sigma = torch.mm(rand, rand.t()) + torch.eye(n)
# sigma = 0.5*(sigma + sigma.t())
# assert ( sigma == sigma.t() ).all()
#
# c = torch.randn(n).sign()
#
# det = torch.det(sigma)
# inv = torch.inverse(sigma)
# tr = torch.trace(inv)
#
# comp_1 = n/5.0 + det ** (1/n) * ( (0.5-1/math.pi)*tr + 1/math.pi*torch.dot(c, torch.matmul(inv, c)) )
# comp_1 = comp_1.item()
#
# num_samples = 10**6
# ide = torch.eye(n)
# estimate = 0
# print("running non-parallelised estimation")
# for _ in tqdm(range(num_samples)):
# 	z = torch.randn(n).abs()
# 	estimate += math.exp( -0.5 * torch.dot(c*z, torch.matmul(det**(1/n)*inv - ide,c*z)) )
# estimate /= num_samples
#
# comp_0 = math.log(2**n / estimate)
#
# print( ("Estimate", "Bound"))
# print( "non-cholesky:")
# print( (comp_0, comp_1) )
# print( "cholesky:")
# print( complexity(sigma, c, num_samples) )
# print()

## Check kernel compared to random networks

depth = 3
width = 5000
num_train_examples = 5
num_networks = 10**3
alpha = 1.0
seed = 0

random.seed(seed)
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
		model = ResidualNet(depth, width, alpha)
		# model = SimpleNet(depth, width, alpha, residual=False)
		# model = JeremySimpleNet(depth, width)
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
sigma_2 = sanitise(torch.mm(data, data.t()) / data.shape[1])
sigma_3 = sanitise(torch.mm(data, data.t()) / data.shape[1])

for _ in range(depth-1):
	new_sigma = np.zeros((num_train_examples, num_train_examples))

	sigma = sigma_1

	for i in range(num_train_examples):
		for j in range(num_train_examples):
			k_x_x = sigma[i][i]
			k_tilde_x_tilde_x = sigma[j][j]
			cross_product_term = np.sqrt(k_x_x * k_tilde_x_tilde_x)

			mean_k_term = sigma[i][j] / cross_product_term
			mean_k_term = np.sqrt((1-mean_k_term**2))
			mean_k_term += mean_k_term*(np.pi - np.arccos(mean_k_term))
			mean_k_term /= np.pi
			t_sigma_term = cross_product_term * mean_k_term

			new_sigma[i][j] = sigma[i][j] + (alpha**2) * t_sigma_term

	sigma_1 = new_sigma

	sigma_2 = increment_kernel_resnet_my_derivation(sigma_2, alpha)
	sigma_3 = increment_kernel(sigma_3)

sigma_2 = sigma_2.numpy()
sigma_3 = sigma_3.numpy()

diff_1 = np.absolute(sigma_1 - sample_cov).max()
diff_2 = np.absolute(sigma_2 - sample_cov).max()
diff_3 = np.absolute(sigma_3 - sample_cov).max()
mean_diff_1 = np.mean(sigma_1 - sample_cov)
mean_diff_2 = np.mean(sigma_2 - sample_cov)
mean_diff_3 = np.mean(sigma_3 - sample_cov)

print("analytical cov:\n", sigma)
print("max difference 1 (paper method): ", diff_1)
print("max difference 2 (our weird adhoc method): ", diff_2)
print("max difference 3 (original jeremy method): ", diff_3)
print("mean difference 1 (paper method): ", mean_diff_1)
print("mean difference 2 (our weird adhoc method): ", mean_diff_2)
print("mean difference 3 (original jeremy method): ", mean_diff_3)
