#!/usr/bin/env python

"""makemore: a character-based language model which will generate Name like words."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


def main():
    #
    # Parameters
    #

    num_steps = 20000
    num_context = 3
    num_dim_per_char = 10
    batch_size = 64
    num_dim_per_char = num_dim_per_char
    input_dim = num_context * num_dim_per_char
    num_neurons = 200

    #
    # Prepare data
    #

    # Read the raw data, which is a list of names of people.
    words = _read_raw_data("names.txt")

    # Build mapping of characters to indices
    # We need to represent the characters as numerical elements
    index_to_char, char_to_index = _build_char_index_map(words)
    num_characters = len(index_to_char)

    # Split the raw data into training, validation, and testing groups.
    random.shuffle(words)
    num_tr = int(0.8 * len(words))
    num_va = int(0.1 * len(words))

    inputs_tr, targets_tr = _build_model_data(
        words[:num_tr], char_to_index, num_context
    )
    inputs_va, targets_va = _build_model_data(
        words[num_tr : num_tr + num_va], char_to_index, num_context
    )
    inputs_te, targets_te = _build_model_data(
        words[num_tr + num_va :], char_to_index, num_context
    )

    #
    # Construct model
    #

    # The embedding layer is used for encoding the input data in a way such that
    # it can be fed into a neural network.
    # In this case, we use a vector of 2 dimensional floats to represent each unique character.
    C = torch.randn((num_characters, num_dim_per_char))

    # Construct hidden layer #1 (aka neurons / weights)
    W1 = torch.randn((input_dim, num_neurons)) * (5 / 3) / (input_dim) ** 0.5  # Weights
    b1 = torch.randn(num_neurons) * 0.1  # Biases

    # Construct output layer
    W2 = torch.randn((num_neurons, num_characters)) * 0.1  # Weights
    b2 = torch.randn(num_characters) * 0.1  # Biases

    # Batch normalization gain & bias
    bngain = torch.randn((1, num_neurons)) * 0.1 + 1.0
    bnbias = torch.randn((1, num_neurons)) * 0.1

    # All parameters
    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True

    #
    # Forward pass
    #

    # Create a random list of indices for sampling from our inputs in this training step.
    batch_indices = torch.randint(0, inputs_tr.shape[0], (batch_size,))

    inputs = inputs_tr[batch_indices]
    targets = targets_tr[batch_indices]

    emb = C[inputs]  # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors

    # Linear layer 1
    hprebn = embcat @ W1 + b1  # hidden layer pre-activation

    # BatchNorm layer
    bnmeani = 1 / batch_size * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = (
        1 / (batch_size - 1) * (bndiff2).sum(0, keepdim=True)
    )  # note: Bessel's correction (dividing by batch_size-1, not batch_size)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    # Non-linearity
    h = torch.tanh(hpreact)  # hidden layer

    # Linear layer 2
    logits = h @ W2 + b2  # output layer

    # cross entropy loss (same as F.cross_entropy(logits, targets))
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes  # subtract max for numerical stability
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = (
        counts_sum**-1
    )  # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(batch_size), targets].mean()

    #
    # Pytorch Autograd backward pass
    #

    for parameter in parameters:
        parameter.grad = None

    for t in [
        logprobs,
        probs,
        counts,
        counts_sum,
        counts_sum_inv,  # afaik there is no cleaner way
        norm_logits,
        logit_maxes,
        logits,
        h,
        hpreact,
        bnraw,
        bnvar_inv,
        bnvar,
        bndiff2,
        bndiff,
        hprebn,
        bnmeani,
        embcat,
        emb,
    ]:
        t.retain_grad()

    loss.backward()

    #
    # Differentiate gradients by hand
    #

    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(batch_size), targets] = -1.0 / batch_size

    dprobs = 1.0 / probs * dlogprobs

    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)

    dcounts_sum = -1 / (counts_sum) ** 2 * dcounts_sum_inv

    dcounts = counts_sum_inv * dprobs
    dcounts += dcounts_sum

    dnorm_logits = norm_logits.exp() * dcounts

    dlogit_maxes = (dnorm_logits * -1).sum(1, keepdim=True)

    dlogits = dnorm_logits.clone()
    z_logits = torch.zeros_like(logits)
    z_logits[range(z_logits.shape[0]), logits.max(1).indices] = 1
    dlogits += z_logits * dlogit_maxes

    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)

    _cmp_grad("logprobs", dlogprobs, logprobs)
    _cmp_grad("probs", dprobs, probs)
    _cmp_grad("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)
    _cmp_grad("counts_sum", dcounts_sum, counts_sum)
    _cmp_grad("counts", dcounts, counts)
    _cmp_grad("norm_logits", dnorm_logits, norm_logits)
    _cmp_grad("logit_maxes", dlogit_maxes, logit_maxes)
    _cmp_grad("logits", dlogits, logits)
    _cmp_grad("h", dh, h)
    _cmp_grad("W2", dW2, W2)
    _cmp_grad("b2", db2, b2)
    # _cmp_grad('hpreact', dhpreact, hpreact)
    # _cmp_grad('bngain', dbngain, bngain)
    # _cmp_grad('bnbias', dbnbias, bnbias)
    # _cmp_grad('bnraw', dbnraw, bnraw)
    # _cmp_grad('bnvar_inv', dbnvar_inv, bnvar_inv)
    # _cmp_grad('bnvar', dbnvar, bnvar)
    # _cmp_grad('bndiff2', dbndiff2, bndiff2)
    # _cmp_grad('bndiff', dbndiff, bndiff)
    # _cmp_grad('bnmeani', dbnmeani, bnmeani)
    # _cmp_grad('hprebn', dhprebn, hprebn)
    # _cmp_grad('embcat', dembcat, embcat)
    # _cmp_grad('W1', dW1, W1)
    # _cmp_grad('b1', db1, b1)
    # _cmp_grad('emb', demb, emb)
    # _cmp_grad('C', dC, C)


def _cmp_grad(name, gradient, tensor):
    """Checks to see if ``gradient`` is equal to ``tensor``'s computed gradient."""
    ex = torch.all(gradient == tensor.grad).item()
    app = torch.allclose(gradient, tensor.grad)
    maxdiff = (gradient - tensor.grad).abs().max().item()
    print(
        f"{name:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}"
    )


def _read_raw_data(file_path):
    with open(file_path, "r") as f:
        words = f.read().splitlines()

    print(f"Read {len(words)} words: {words[:6]}")
    return words


def _build_char_index_map(words):
    """Utility to create mappings from characters to their index representations"""
    index_to_char = ["."]
    index_to_char.extend(sorted(list(set("".join(words)))))
    char_to_index = {char: index for index, char in enumerate(index_to_char)}
    return index_to_char, char_to_index


def _build_model_data(words, char_to_index, num_context):
    inputs = []  # inputs to the neural net.
    targets = []  # target/label for each data point in inputs.

    for w in words[:]:
        # Initialize context to "..." (expressed in indices)
        context = [char_to_index["."]] * num_context

        # Fill inputs & targets by scanning each character in the word.
        for char in w + ".":
            char_index = char_to_index[char]
            inputs.append(context)
            targets.append(char_index)

            # Slide context cursor "forwards"
            context = context[1:] + [char_index]

    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)

    return inputs, targets


if __name__ == "__main__":
    main()
