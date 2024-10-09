"""
Creating a self attention function, and verifying that it reproduces
PyTorch's output.

ver2: supports minibatching.

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
    K, Q, V = np.split(X@W_KQV, 3, axis=-1)
    dim = X.shape[-1]
    softmax_out = softmax(K@Q.swapaxes(-1, -2)/np.sqrt(dim) + mask, axis=-1)
    return softmax_out@V@W_out


B = 50  # Number of batches
T = 100  # Number of tokens in input (the transformer can, in principle accept token sequences of any length)
d = 64  # Model dimension. Querry and Key dimensions are d/heads.

torch_attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)

# mask is uppertriangular matrix of -inf
mask = torch.triu(-float("inf")*torch.ones(T, T), 1)

# Trial input
X = torch.randn(B, T, d)

# PyTorch output:
torch_Y, torch_A = torch_attn(X, X, X, attn_mask=mask)

# Weights used by PyTorch:
W_KQV = torch_attn.in_proj_weight.detach().numpy().T
W_out = torch_attn.out_proj.weight.detach().numpy().T

# My output:
my_Y = my_attn(X.numpy(), mask.numpy(), W_KQV, W_out)

# Comparing my output to PyTorch output:
print(np.linalg.norm(my_Y - torch_Y.detach().numpy()))
