"""
Creating a self attention function, and verifying that it reproduces
PyTorch's output.

ver1: no minibatching.

Credit to Youtube Channel @deeplearningsystemscourse1116 for 
reference code.

"""

import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn


def my_attn(X, mask, W_KQV, W_out):
    """
    Parameters:
        W_KQV: array of weights, dimension?
    """
    K, Q, V = np.split(X@W_KQV, 3, axis=1)
    dim = X.shape[1]
    softmax_out = softmax(K@Q.T/np.sqrt(dim) + mask, axis=-1)
    return softmax_out@V@W_out


# Dimensions
T, d = 100, 64

torch_attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)

# mask is uppertriangular matrix of -inf
mask = torch.triu(-float("inf")*torch.ones(T, T), 1)

# Trial input
X = torch.randn(1, T, d)

# PyTorch output:
torch_Y, torch_A = torch_attn(X, X, X, attn_mask=mask)

# Weights used by PyTorch:
W_KQV = torch_attn.in_proj_weight.detach().numpy().T
W_out = torch_attn.out_proj.weight.detach().numpy().T

# My output:
my_Y = my_attn(X[0].numpy(), mask.numpy(), W_KQV, W_out)

# Comparing my output to PyTorch output:
print(np.linalg.norm(my_Y - torch_Y[0].detach().numpy()))
