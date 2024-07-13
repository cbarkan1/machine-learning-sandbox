"""
Creating a full transformer, and verifying that it reproduces
PyTorch's output.

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


def layer_norm(Z, eps):
    return (Z - Z.mean(axis=-1, keepdims=True)) / np.sqrt(
        Z.var(axis=-1, keepdims=True) + eps)


def relu(Z):
    return np.maximum(Z, 0)


def my_transformer(X, mask, head, W_KQV, W_out, W_ff1, W_ff2, eps):
    """
    Parameters:
        W_ff1: weights for first feed forward part
        W_ff2: weights for second feed forward part
    """
    attn = my_attn(X, mask, heads, W_KQV, W_out)
    Z = layer_norm(X + attn, eps)
    return layer_norm(Z + relu(Z@W_ff1)@W_ff2, eps)


B = 50  # Number of batches
T = 100  # Model dimension?
d = 64  # Querry and Key dimension
heads = 4


torch_transformer = nn.TransformerEncoderLayer(
    d, heads, dim_feedforward=128, dropout=0., batch_first=True)
torch_transformer.linear1.bias.data.zero_()
torch_transformer.linear2.bias.data.zero_()


# mask is uppertriangular matrix of -inf
mask = torch.triu(-float("inf")*torch.ones(T, T), 1)

# Trial input
X = torch.randn(B, T, d)

# PyTorch output:
torch_Y = torch_transformer(X, mask)

# Weights used by PyTorch:
W_KQV = torch_transformer.self_attn.in_proj_weight.detach().numpy().T
W_out = torch_transformer.self_attn.out_proj.weight.detach().numpy().T
W_ff1 = torch_transformer.linear1.weight.detach().numpy().T
W_ff2 = torch_transformer.linear2.weight.detach().numpy().T
eps = torch_transformer.norm1.eps

# My output:
my_Y = my_transformer(X.numpy(), mask.numpy(), heads, W_KQV, W_out, 
                      W_ff1, W_ff2, eps)

# Comparing my output to PyTorch output:
print(np.linalg.norm(my_Y - torch_Y.detach().numpy()))
