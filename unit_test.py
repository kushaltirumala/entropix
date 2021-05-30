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
seed = 0
device = None
if torch.cuda.is_available():
	device = torch.device("cuda:1")
else:
	device = torch.device("cpu")


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

jeremy_baseline_method_max_abs = []
jeremy_baseline_method_mean = []
out_baseline_method_max_abs = []
out_baseline_method_mean = []

for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:

	_, _, train_loader, _ = get_data( num_train_examples=num_train_examples,
									  num_test_examples=None,
	                                  batch_size=num_train_examples,
		                              random_labels=False,
		                              binary_digits=False )

	for data, target in train_loader:
	    data, target = normalize_data(data, target)

	data = data.to(device)
	target = target.to(device)

	out_matrix = torch.from_numpy(np.zeros((num_train_examples, num_networks))).to(device)

	with torch.no_grad():
		print(f"Sampling {num_networks} random networks")
		for network_idx in tqdm(range(num_networks)):
			model = ResidualNet(depth, width, alpha)
			# model = SimpleNet(depth, width, alpha, residual=False)
			# model = JeremySimpleNet(depth, width)
			model = model.to(device)
			for p in model.parameters():
				p.data = torch.randn_like(p) / math.sqrt(p.shape[1])

			pred = model(data).squeeze()
			out_matrix[:, network_idx] = pred

	sample_mean = torch.mean(out_matrix, dim=1)
	sample_cov = torch.from_numpy(np.cov(out_matrix.cpu()))

	print()
	print("sample mean:\n", sample_mean)
	print("sample cov:\n", sample_cov)


	sigma_1 = sanitise((torch.mm(data, data.t()) / data.shape[1]))
	# sigma_2 = sanitise((torch.mm(data, data.t()) / data.shape[1]))
	sigma_3 = sanitise((torch.mm(data, data.t()) / data.shape[1]))

	for _ in range(depth-1):
		new_sigma = torch.zeros((num_train_examples, num_train_examples))

		sigma = sigma_1


		for i in range(num_train_examples):
			for j in range(num_train_examples):
				k_x_x = sigma[i][i]
				k_tilde_x_tilde_x = sigma[j][j]
				cross_product_term = torch.sqrt(k_x_x * k_tilde_x_tilde_x)

				mean_k_term = sigma[i][j] / cross_product_term

				h_mean_k_term = (1.0/math.pi) * (math.sqrt(1 - mean_k_term**2) + mean_k_term * (math.pi - torch.acos(mean_k_term)))

				t_sigma_term = cross_product_term * h_mean_k_term

				new_sigma[i][j] = ((1 - alpha**2)) * sigma[i][j] + (alpha**2) * t_sigma_term


		sigma_1 = new_sigma

		# sigma_2 = increment_kernel_resnet_my_derivation(sigma_2, alpha)
		sigma_3 = increment_kernel(sigma_3)

	sample_cov = sample_cov.to(device)
	sigma_1 = sigma_1.to(device)
	# sigma_2 = sigma_2.to(device)
	sigma_3 = sigma_3.to(device)

	diff_1 = torch.max(torch.abs(sigma_1 - sample_cov))
	# diff_2 = torch.max(torch.abs(sigma_2 - sample_cov))
	diff_3 = torch.max(torch.abs(sigma_3 - sample_cov))
	mean_diff_1 = torch.mean(torch.abs(sigma_1 - sample_cov))
	# mean_diff_2 = torch.mean(torch.abs(sigma_2 - sample_cov))
	mean_diff_3 = torch.mean(torch.abs(sigma_3 - sample_cov))

	print("sample cov:\n", sample_cov)
	print("analytical cov sigma 1:\n", sigma_1)
	# print("analytical cov sigma 2:\n", sigma_2)
	print("analytical cov sigma 3:\n", sigma_3)
	print("max difference 1 (paper method): ", diff_1.item())
	# print("max difference 2 (our weird adhoc method): ", diff_2.item())
	print("max difference 3 (original jeremy method): ", diff_3.item())
	print("mean difference 1 (paper method): ", mean_diff_1.item())
	# print("mean difference 2 (our weird adhoc method): ", mean_diff_2.item())
	print("mean difference 3 (original jeremy method): ", mean_diff_3.item())
	jeremy_baseline_method_max_abs.append(diff_1.item())
	jeremy_baseline_method_mean.append(diff_3.item())
	out_baseline_method_max_abs.append(mean_diff_1.item())
	out_baseline_method_mean.append(mean_diff_3.item())

jeremy_baseline_method_max_abs = np.array(jeremy_baseline_method_max_abs)
jeremy_baseline_method_mean = np.array(jeremy_baseline_method_mean)
out_baseline_method_max_abs = np.array(out_baseline_method_max_abs)
out_baseline_method_mean = np.array(out_baseline_method_mean)

np.save(open("jeremy_baseline_method_max_abs_wrong.npy", "wb"), jeremy_baseline_method_max_abs)
np.save(open("jeremy_baseline_method_mean_wrong.npy", "wb"), jeremy_baseline_method_mean)
np.save(open("out_baseline_method_max_abs_wrong.npy", "wb"), out_baseline_method_max_abs)
np.save(open("out_baseline_method_mean_wrong.npy", "wb"), out_baseline_method_mean)
