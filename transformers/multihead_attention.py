"""
Creating a multihead self attention function, and verifying that it
reproduces PyTorch's output.

Credit to Youtube Channel @deeplearningsystemscourse1116 for 
reference code.

"""

import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn


def my_attn(X, mask, heads, W_KQV, W_out):
    """
    Parameters:
        W_KQV: array of weights, dimension?
    """
    B, T, d = X.shape
    K, Q, V = np.split(X@W_KQV, 3, axis=-1)

    # Reshaping K, Q, V so that they are B x heads x T x d/heads
    K = K.reshape(B, T, heads, d//heads).swapaxes(1, 2)
    Q = Q.reshape(B, T, heads, d//heads).swapaxes(1, 2)
    V = V.reshape(B, T, heads, d//heads).swapaxes(1, 2)

    softmax_out = softmax(K@Q.swapaxes(-1, -2)/np.sqrt(d//heads) 
                          + mask, axis=-1)
    return (softmax_out@V).swapaxes(1, 2).reshape(B, T, d) @ W_out


B = 50  # Number of batches
T = 100  # Number of tokens in input (the transformer can, in principle accept token sequences of any length)
d = 64  # Model dimension. Querry and Key dimensions are d/heads.
heads = 4

torch_attn = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)

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
my_Y = my_attn(X.numpy(), mask.numpy(), heads, W_KQV, W_out)

# Comparing my output to PyTorch output:
print(np.linalg.norm(my_Y - torch_Y.detach().numpy()))
