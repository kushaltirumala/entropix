import math
import torch
import numpy as np

def sanitise(sigma):
    return torch.clamp(sigma, -1, 1)
    # return sigma.clamp(min=-1,max=1)

def increment_kernel(sigma):
    # new_sigma = torch.sqrt(1-sigma**2)
    # new_sigma += sigma*(math.pi - torch.acos(sigma))
    # new_sigma /= math.pi
    # return sanitise(new_sigma)

    new_sigma = torch.zeros((sigma.shape[0], sigma.shape[0]))

    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[0]):
            k_x_x = sigma[i][i]
            k_tilde_x_tilde_x = sigma[j][j]
            cross_product_term = torch.sqrt(k_x_x * k_tilde_x_tilde_x)

            mean_k_term = sigma[i][j] / cross_product_term

            new_mean_k_term = torch.sqrt(1-mean_k_term**2)
            new_mean_k_term += mean_k_term*(math.pi - torch.acos(mean_k_term))
            new_mean_k_term /= math.pi
            t_sigma_term = cross_product_term * new_mean_k_term

            new_sigma[i][j] = t_sigma_term

    return sanitise(new_sigma)


def increment_kernel_resnet_my_derivation(sigma, alpha):
    new_sigma = torch.sqrt(1-sigma**2)
    new_sigma += sigma*(math.pi - torch.acos(sigma))
    new_sigma /= math.pi

    new_sigma *= (1 + alpha**2)

    return sanitise(new_sigma)
# TODO: rewrite with cholesky_solve and Frobenius norm trace
def complexity(sigma, c, samples):
    n = sigma.shape[0]
    device = sigma.device
    id = torch.eye(n, device=device)

    assert ( sigma == sigma.t() ).all()
    u = torch.cholesky(sigma)
    inv = torch.cholesky_inverse(u)
    assert (torch.mm(sigma, inv) - id).abs().max() < 1e-03

    nth_root_det = u.diag().pow(2/n).prod()
    inv_trace = inv.diag().sum()
    inv_proj = torch.dot(c, torch.mv(inv, c))

    formula_1 = n/5.0 + nth_root_det * ( (0.5 - 1/math.pi)*inv_trace + inv_proj/math.pi )

    mat = nth_root_det*inv - id

    z = torch.randn((n,samples), device=device).abs()
    cz = torch.mul(c.unsqueeze(1), z)
    contribs = -0.5*(cz * torch.matmul(mat, cz)).sum(dim=0)

    formula_0 = n*math.log(2) + math.log(samples) - torch.logsumexp(contribs, dim=0)

    return formula_0.item(), formula_1.item()

def invert_bound(x):
    return 1-math.exp(-x)
