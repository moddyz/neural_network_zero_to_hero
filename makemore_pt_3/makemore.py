#!/usr/bin/env python

"""makemore: a character-based language model which will generate Name like words."""

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def read_raw_data():
    with open("names.txt", "r") as f:
        words = f.read().splitlines()

    print(f"Read {len(words)} words: {words[:6]}")

    return words


def build_char_index_mapping(words):
    index_to_char = ["."]
    index_to_char.extend(sorted(list(set(''.join(words)))))
    char_to_index = {char: index for index, char in enumerate(index_to_char)}
    return index_to_char, char_to_index


def build_network_data(words, char_to_index, num_context):
    inputs = [] # inputs to the neural net.
    targets = [] # target/label for each data point in inputs.

    for w in words[:]:

        # Initialize context to "..." (expressed in indices)
        context = [char_to_index["."]] * num_context

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


def build_network_data_sets(words, char_to_index, num_context):
    # Partition into training, validation, and test data sets.
    random.seed(42)
    random.shuffle(words)
    num_tr = int(0.8*len(words))
    num_va = int(0.1*len(words))

    inputs_tr, targets_tr = build_network_data(words[:num_tr], char_to_index, num_context)
    inputs_va, targets_va = build_network_data(words[num_tr:num_tr + num_va], char_to_index, num_context)
    inputs_te, targets_te = build_network_data(words[num_tr + num_va:], char_to_index, num_context)

    return inputs_tr, targets_tr, inputs_va, targets_va, inputs_te


def main():
    # How many characters should we keep in context to predict the next char?
    num_context = 3

    # Read the raw data, which is a list of names of people.
    words = read_raw_data()

    # Build mapping of characters to indices
    # We need to represent the characters as numerical elements
    index_to_char, char_to_index = build_char_index_mapping(words)
    num_characters = len(index_to_char)

    # Partition the data into 3 groups:
    # - Training
    # - Validation
    # - Testing
    #
    # Each entry in the inputs will look something like "emm" where the corresponding target element
    # is the character after "emm", for example, "a".
    #
    # Since our context is 3 char wide, we use "." to represent an "blank" character to fill in empty space before or
    # after the word.
    inputs_tr, targets_tr, inputs_va, targets_va, inputs_te = build_network_data_sets(words, char_to_index, num_context)

    # Seed the random generator.
    g = torch.Generator().manual_seed(2147483647)

    # The embedding layer is used for encoding the input data in a way such that
    # it can be fed into a neural network.
    # In this case, we use a vector of 2 dimensional floats to represent each unique character.
    num_dim_per_char = 2
    C = torch.randn((num_characters, num_dim_per_char), generator=g)

    #
    # Construct hidden layer #1 (aka neurons / weights)
    #

    # Because each
    per_input_dim = num_context * num_dim_per_char

    num_neurons = 200
    W1 = torch.randn((per_input_dim, num_neurons), generator=g) # Weights
    b1 = torch.randn(num_neurons, generator=g) # Biases

    # Construct output layer
    W2 = torch.randn((num_neurons, num_characters), generator=g) * 0.1 # Weights
    b2 = torch.randn(num_characters, generator=g) * 0 # Biases

    # All parameters
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    num_steps = 200000
    batch_size = 32
    losses = []
    for step in range(num_steps):

        # Create a random list of indices for sampling from our inputs in this training step.
        batch_indices = torch.randint(0, inputs_tr.shape[0], (batch_size,), generator=g)

        # Extract
        inputs_tr[batch_indices]

        # Forward pass

        inputs_emb = C[inputs_tr[batch_indices]]
        h = torch.tanh(inputs_emb.view(inputs_emb.shape[0], per_input_dim) @ W1 + b1)

        plt.hist(h.view(-1).tolist(), 50)
        plt.show()
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, targets_tr[batch_indices]) # computes log likelihood for classification purposes

        losses.append(loss.item())

        # Backward pass

        # Reset gradient first.
        for parameter in parameters:
            parameter.grad = None

        # Calculate gradients for each parameter.
        loss.backward()

        # Update parameter data.
        learning_rate = 0.01 if step < 100000 else 0.001
        for parameter in parameters:
            parameter.data += -learning_rate * parameter.grad

        if step % 10000 == 0:
            print(f"{step:7d} / {num_steps:7d}: {loss.item()}")

        break

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
        context = [char_to_index["."]] * num_context
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


if __name__ == "__main__":
    main()

