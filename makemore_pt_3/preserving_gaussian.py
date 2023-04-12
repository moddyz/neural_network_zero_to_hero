#!/usr/bin/env python

"""Example of multiplying the weights layer by the square root of
the # of dimensions per input to preserve the standard deviation of
the gaussian distribution"""

import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":

    input_count = 1000
    dims_per_input = 10
    neuron_count = 200

    inputs = torch.randn(input_count, dims_per_input)
    weights = torch.randn(dims_per_input, neuron_count) / (dims_per_input**0.5)
    preactivation = inputs @ weights

    print(f"{inputs.mean()}, {inputs.std()}")
    print(f"{preactivation.mean()}, {preactivation.std()}")

    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.hist(inputs.view(-1).tolist(), 50, density=True)
    plt.subplot(122)
    plt.hist(preactivation.view(-1).tolist(), 50, density=True)
    plt.show()


