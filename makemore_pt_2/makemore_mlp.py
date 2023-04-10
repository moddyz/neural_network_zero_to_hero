#!/usr/bin/env python

"""Training a character-based language model with more context on previous chars."""

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def build_data_set(words, char_to_index, block_size):
    # Build the input training data set.
    inputs = [] # inputs to the neural net.
    targets = [] # target/label for each data point in inputs.

    for w in words[:]:

        # Initialize context to "..." (expressed in indices)
        context = [char_to_index["."]] * block_size

        # Fill inputs & targets by scanning each character in the word.
        for char in (w + "."):
            char_index = char_to_index[char]
            inputs.append(context)
            targets.append(char_index)

            # Slide context cursor "forwards"
            context = context[1:] + [char_index]

    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)

    return inputs, targets

if __name__ == "__main__":

    # Read input words
    with open("names.txt", "r") as f:
        words = f.read().splitlines()
    print(f"Read {len(words)} words: {words[:6]}")

    # Build index <-> character mappings
    index_to_char = ["."]
    index_to_char.extend(sorted(list(set(''.join(words)))))
    num_chars = len(index_to_char)
    char_to_index = {char: index for index, char in enumerate(index_to_char)}

    # Partition into training, validation, and test data sets.
    random.seed(42)
    random.shuffle(words)
    num_tr = int(0.8*len(words))
    num_va = int(0.1*len(words))

    block_size = 3 # how many characters should we keep in context to predict the next char?
    inputs_tr, targets_tr = build_data_set(words[:num_tr], char_to_index, block_size)
    inputs_va, targets_va = build_data_set(words[num_tr:num_tr + num_va], char_to_index, block_size)
    inputs_te, targets_te = build_data_set(words[num_tr + num_va:], char_to_index, block_size)

    g = torch.Generator().manual_seed(2147483647)

    # Cram num of different characters into a 2 dimensional space.
    num_input_dims = 10
    C = torch.randn((num_chars, num_input_dims), generator=g, requires_grad=True)

    # Construct hidden layer #1 (aka neurons / weights)
    per_input_dim = block_size * num_input_dims # Number of values encoding a single input row.
    num_neurons = 300
    W1 = torch.randn((per_input_dim, num_neurons), generator=g, requires_grad=True) # Weights
    b1 = torch.randn(num_neurons, generator=g, requires_grad=True) # Biases

    # Construct output layer
    W2 = torch.randn((num_neurons, num_chars), generator=g, requires_grad=True) # Weights
    b2 = torch.randn(num_chars, generator=g, requires_grad=True) # Biases

    # All parameters
    parameters = [C, W1, b1, W2, b2]

    num_steps = 100000
    batch_size = 256
    learning_rates = torch.linspace(0.1, 0.0001, num_steps)
    losses = []
    for step in range(num_steps):

        print(f"Training iteration #{step}")

        # mini batch
        batch_indices = torch.randint(0, inputs_tr.shape[0], (batch_size,), generator=g)

        # Forward pass

        inputs_emb = C[inputs_tr[batch_indices]]
        h = torch.tanh(inputs_emb.view(inputs_emb.shape[0], per_input_dim) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, targets_tr[batch_indices]) # computes log likelihood for classification purposes

        print(f"{loss.item()}")
        losses.append(loss.item())

        # Backward pass

        # Reset gradient first.
        for parameter in parameters:
            parameter.grad = None

        # Calculate gradients for each parameter.
        loss.backward()

        # Update parameter data.
        learning_rate = learning_rates[step]
        for parameter in parameters:
            parameter.data += -learning_rate * parameter.grad

    # Compute training loss.
    emb_tr = C[inputs_tr]
    h_tr = torch.tanh(emb_tr.view(emb_tr.shape[0], per_input_dim) @ W1 + b1)
    logits_tr = h_tr @ W2 + b2
    loss_tr = F.cross_entropy(logits_tr, targets_tr) # computes log likelihood for classification purposes
    print(f"Training loss: {loss_tr}")

    emb_va = C[inputs_va]
    h_va = torch.tanh(emb_va.view(emb_va.shape[0], per_input_dim) @ W1 + b1)
    logits_va = h_va @ W2 + b2
    loss = F.cross_entropy(logits_va, targets_va) # computes log likelihood for classification purposes
    print(f"Validation loss: {loss}")

    plt.plot(range(num_steps), losses)
    plt.show()

    # Sample from model
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(20):
        word = ""
        context = [char_to_index["."]] * block_size
        while True:
            emb_sample = C[torch.tensor([context])]
            h_sample = torch.tanh(emb_sample.view(emb_sample.shape[0], per_input_dim) @ W1 + b1)
            logits_sample = h_sample @ W2 + b2
            probs = F.softmax(logits_sample, dim=1)
            char_index = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [char_index]
            word += index_to_char[char_index]

            if char_index == char_to_index["."]:
                break

        print(word)
