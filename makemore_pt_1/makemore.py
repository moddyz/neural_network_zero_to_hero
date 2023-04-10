#!/usr/bin/env python

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def draw(N, index_to_char):
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    for y in range(27):
        for x in range(27):
            char_pair = index_to_char[y] + index_to_char[x]
            plt.text(x, y, char_pair, ha="center", va="bottom", color="gray", fontsize=7)
            plt.text(x, y, N[y, x].item(), ha="center", va="top", color="gray", fontsize=7)

    plt.axis("off")
    plt.show()


def sample_name(P, index_to_char, g):
    cur_idx = 0
    word = ''
    while True:
        p = P[cur_idx]
        next_idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        next_char = index_to_char[next_idx]
        if next_char == ".":
            break
        word += next_char
        cur_idx = next_idx

    return word



def loss_function(N):
    P = N.float()
    P = P / P.sum(1, keepdim=True)

    g = torch.Generator().manual_seed(2147483646)
    for x in range(10):
        print(sample_name(P, index_to_char, g))

    log_likelihood = 0.0
    n = 0
    for w in words[:3]:
        chars = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chars, chars[1:]):

            i_x = char_to_index[ch1]
            i_y = char_to_index[ch2]
            prob = P[i_x, i_y]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
            print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

    print(f"{log_likelihood=}")
    nll = -log_likelihood
    print(f"{nll=}")
    nnll = nll / n
    print(f"{nnll=}")


if __name__ == "__main__":
    with open("names.txt", "r") as f:
        words = f.read().splitlines()

    index_to_char = ["."]
    index_to_char.extend(sorted(list(set(''.join(words)))))
    char_to_index = {char: index for index, char in enumerate(index_to_char)}

    # Create inputs and targets based on words.
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

    print(num_inputs, targets.shape)



    for num_pass in range(100):

        #
        # Forward pass
        #

        # Calculate probabilities
        logits = (inputs_enc @ W)
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        target_probs = probs[torch.arange(num_inputs), targets]
        print(target_probs.shape)
        log_target_probs = target_probs.log()
        loss = -log_target_probs.mean()

        # Backward pass
        W.grad = None
        loss.backward()

        W.data += -50 * W.grad

        print(loss)

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

        print(word)
