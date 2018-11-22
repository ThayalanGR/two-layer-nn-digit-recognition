
# NOTE: You can only use Tensor API of PyTorch

import torch
import numpy as np

# Extra TODO: Document with proper docstring


def sigmoid(z):
    """Calculates sigmoid values for tensors

    """
    result = (1 / (1 + np.exp(-z)))
    return result

# Extra TODO: Document with proper docstring


def delta_sigmoid(z):
    """Calculates derivative of sigmoid function

    """
    grad_sigmoid = z * (1 - z)
    return grad_sigmoid

# Extra TODO: Document with proper docstring


def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors

    """
    
    exp_scores = np.exp(x)
    stable_softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return stable_softmax


if __name__ == "__main__":
    pass
