# sigmoid, sigmoid_backward, relu, relu_backward

import numpy as np

# -----------------------------------------------------------------------------------------------------------------------------

def sigmoid(Z):
    """
    Implements the sigmoid function.

    Returns: 
    a --> activation
    activation_cache --> contains Z value
    """

    s = 1 / (1 + np.exp(-np.array(Z)))

    A = s
    activation_cache = Z

    return A, activation_cache

def relu(Z):
    """
    Implements the relu function.

    Returns: 
    a -- activation
    activation_cache -- contains Z value
    """

    r = np.where(Z > 0, Z, 0)
    A = r
    activation_cache = Z

    return A, activation_cache

def sigmoid_backward(dA, activation_cache):
    """
    Returns dZ.
    """
    s, _ = sigmoid(activation_cache)
    dS = np.multiply(s, np.subtract(1, s))
    dZ = np.multiply(dA, dS)

    return dZ

def relu_backward(dA, activation_cache):
    """
    Return dZ
    """

    dR = np.where(activation_cache > 0, 1, 0)
    dZ = np.multiply(dA, dR)

    return dZ

print("dnn_utils_v1.py compiled successfully!")
