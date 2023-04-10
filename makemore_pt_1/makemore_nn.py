#!/usr/bin/env python

"""makemore part 1 code implementing a bigram model using a neural network"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


if __name__ == "__main__":

    #
    # Prepare training data
    #

    # Read input data
    with open("names.txt", "r") as f:
        words = f.read().splitlines()

    # Build index <-> character mappings
    index_to_char = ["."]
    index_to_char.extend(sorted(list(set(''.join(words)))))
    char_to_index = {char: index for index, char in enumerate(index_to_char)}

    # Create inputs and targets tensors based on words.
    inputs = []
    targets = []
    for w in words:
        chars = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chars, chars[1:]):

            input_idx = char_to_index[ch1]
            target_idx = char_to_index[ch2]

            inputs.append(input_idx)
            targets.append(target_idx)

    # Convert to tensors
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)

    # Get number of training data.
    num_inputs = inputs.nelement()

    # Convert to one-hot-encoding
    inputs_enc = F.one_hot(inputs, num_classes=27).float()
    targets_enc = F.one_hot(targets, num_classes=27).float()

    # Initialize neural network layer
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    #
    # Prepare our model
    #

    for num_pass in range(100):

        print(f"Training iteration: {num_pass}")

        # Forward pass
        logits = (inputs_enc @ W)
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        target_probs = probs[torch.arange(num_inputs), targets]
        log_target_probs = target_probs.log()
        loss = -log_target_probs.mean()

        print(f"\tComputed loss: {loss}")

        # Backward pass
        W.grad = None
        loss.backward()
        # Gradient descent
        W.data += -50 * W.grad

    #
    # Sample from our model to generate new words like the training set
    #

    g = torch.Generator().manual_seed(2147483647)

    for i in range(5):

        # Start with "."
        sample_index = 0
        word = ''
        while True:
            # Extract the
            sample_input_enc = F.one_hot(torch.tensor([sample_index]), num_classes=27).float()
            logits = sample_input_enc @ W
            counts = logits.exp()
            sample_probs = counts / counts.sum(1, keepdim=True)

            # Sample based on probability distribution.
            sample_index = torch.multinomial(sample_probs, num_samples=1, replacement=True, generator=g).item()
            sample_char = index_to_char[sample_index]
            if sample_index == 0:
                break
            word += sample_char

        print(f"Sampled word: {word}")
