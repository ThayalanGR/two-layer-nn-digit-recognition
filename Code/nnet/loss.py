
# NOTE: You can only use Tensor API of PyTorch

import torch
import numpy as np
from nnet import activation
# Extra TODO: Document with proper docstring


def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector. 
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    x = outputs
    y = labels
    m = y.shape[0]
    p = activation.softmax(x)
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    log_likelihood = -np.log(p[range(m), y])
    creloss = np.sum(log_likelihood) / m
    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring


def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector. 
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    x = outputs
    y = labels
    m = y.shape[0]
    grad = activation.softmax(x)
    grad[range(m), y] -= 1
    avg_grads = grad/m
    return avg_grads


if __name__ == "__main__":
    pass
