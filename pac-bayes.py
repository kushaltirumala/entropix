import math
import torch
import pickle
import itertools
import numpy as np

from util.data import get_data, normalize_data
from util.kernel import sanitise, increment_kernel, complexity, invert_bound, increment_kernel_resnet_my_derivation
from util.trainer import train_network

from tqdm import tqdm



### Dependent variables
num_train_examples_list = [784]
# num_train_examples_list = [500]

alpha_list = [0, 0.001, 0.01, 0.1, 0.2, 0.3]
# random_labels_list = [True, False]
# binary_digits_list = [True, False]
depth_list = [2]
width_list = [784]
seed_list = [0]

### Data hyperparameters
batch_size = 784

### Training hyperparameters
init_lr = 0.01
decay = 0.9

### Estimator hyperparameters
num_samples = 10**6
cuda = False
delta = 0.01

# param_product = itertools.product( num_train_examples_list, depth_list, width_list, seed_list, alpha_list )
# for params in tqdm(param_product):
# try:
num_train_examples, depth, width, seed = 500, 34, 64, 0

### Set random seed
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

### Get data
full_batch_train_loader, _, train_loader, test_loader = get_data( num_train_examples=num_train_examples,
                                                                  num_test_examples=None,
                                                                  batch_size=batch_size,
                                                                  random_labels=False,
                                                                  binary_digits=False )

### Train network
# train_acc, test_acc, model = train_network( train_loader = train_loader,
#                                             test_loader = test_loader,
#                                             depth=depth,
#                                             width=width,
#                                             init_lr=init_lr,
#                                             decay=decay,
#                                             cuda=cuda,
#                                             alpha=alpha )

# print(f"Train acc: {train_acc[-1]}")
# print(f"Test acc: {test_acc}")

### Kernel arithmetic
data, target = list(full_batch_train_loader)[0]
if cuda: data, target = data.cuda(), target.cuda()
data, target = normalize_data(data, target)

c = target.float()

sigma = sanitise(torch.mm(data, data.t()) / data.shape[1])

assert ( sigma == sigma.t() ).all()
n = sigma.shape[0]

for _ in range(depth-1):
	new_sigma = torch.zeros((num_train_examples, num_train_examples))

	for i in range(num_train_examples):
		for j in range(num_train_examples):
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
	sigma = new_sigma

    # import pdb; pdb.set_trace()
    # c0_1, c1 = complexity( sigma, c, num_samples )
    # c0_2, c1 = complexity( sigma, c, num_samples )
    # c0_3, c1 = complexity( sigma, c, num_samples )
    #
    #
    # delta_term = math.log(2*n/delta)
    #
    # estimator_1 = invert_bound( (c0_1 + delta_term) / (n-1) )
    # estimator_2 = invert_bound( (c0_2 + delta_term) / (n-1) )
    # estimator_3 = invert_bound( (c0_3 + delta_term) / (n-1) )
    #
    # bound = invert_bound( (c1 + delta_term) / (n-1) )
    #
    # print(f"Estimator 1: {estimator_1}")
    # print(f"Estimator 2: {estimator_2}")
    # print(f"Estimator 3: {estimator_3}")
    # print(f"Bound: {bound}")
    #
    # results = (train_acc, test_acc, estimator_1, estimator_2, estimator_3, bound, c0_1, c0_2, c0_3, c1)
    # fname = 'logs_run_1/pac-bayes/' + str(params) + '.pickle'
    # pickle.dump( results, open( fname, "wb" ) )
    # except RuntimeError e:
    #     print("errored out on param config")
    #     print(params)
